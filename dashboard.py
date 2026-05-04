"""
EEG Dashboard — fullscreen passive monitoring.

Two core metrics from frontal channels FP1 (ch0) / FP2 (ch1):
  • Mental state  — Focus / Relax / Neutral via α / β balance gauge
  • Stress (FAA)  — Frontal Alpha Asymmetry on a bipolar gauge
                    Left = stress (FP1 dominant), Right = calm (FP2 dominant)

Features:
  • Personal calibration baseline (press C — 10-second eyes-open rest)
  • α/β bipolar balance gauge
  • FAA bipolar gauge (personal or population baseline)
  • 60-second rolling state + stress timeline
  • Q or Escape to quit

Flags:
  --no-gui   headless: prints one-line metrics per second to stdout
  --log FILE also write headless output to FILE
"""

import argparse
import sys
import numpy as np
import time
import threading
from collections import deque

from pylsl import resolve_streams, StreamInlet
from scipy.signal import butter, filtfilt, welch

# ---------------------------------------------------------------------------
# Theme  (resolved after pygame init)
# ---------------------------------------------------------------------------

W, H = 0, 0

BG        = (242, 243, 246)
PANEL     = (255, 255, 255)
BORDER    = (200, 203, 212)
TEXT      = (12,  14,  22)
SUBTEXT   = (55,  60,  78)
TRACK     = (215, 218, 226)
FOCUS_C   = (37,  99,  235)
RELAX_C   = (22,  163, 74)
NEUTRAL_C = (100, 105, 120)
STRESS_H  = (205, 50,  40)
STRESS_L  = (22,  163, 74)
STRESS_M  = (217, 119, 6)
ALPHA_C   = (67,  110, 215)
BETA_C    = (200, 60,  50)

screen = None
FONT = SMALL = TITLE = TINY = BIG = None


def _sysfont(size, bold=False):
    """Return the best available system sans-serif font at the given pixel size."""
    import pygame
    for name in ("helveticaneue", "helvetica", "sfnsdisplay", "sfprodisplay",
                 "segoeui", "inter", "roboto", "arial", ""):
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f:
                return f
        except Exception:
            pass
    return pygame.font.Font(None, size)


def _init_gui():
    global screen, FONT, SMALL, TITLE, TINY, BIG, W, H
    import pygame
    import sys

    # Make DPI-aware on Windows to fix fullscreen scaling issues
    if sys.platform == 'win32':
        try:
            import ctypes
            # Tell Windows this app is DPI-aware
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
        except Exception:
            pass

    pygame.init()
    # Pass (0, 0) so pygame picks the native display resolution automatically.
    # Querying display.Info() *before* set_mode can return wrong values on
    # HiDPI / scaled displays, causing content to be cropped in fullscreen.
    # DPI awareness (set above) ensures pygame gets correct dimensions.
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H   = screen.get_width(), screen.get_height()
    pygame.display.set_caption("EEG Dashboard")
    s = H / 900
    BIG   = _sysfont(int(54 * s), bold=True)
    TITLE = _sysfont(int(20 * s), bold=True)
    FONT  = _sysfont(int(16 * s), bold=True)
    SMALL = _sysfont(int(14 * s), bold=False)
    TINY  = _sysfont(int(12 * s), bold=False)
    return pygame


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class EEGDashboard:

    CALIB_DURATION = 10.0   # seconds

    def __init__(self):
        self.connection_status = "Connecting…"
        self.running = True

        self.fs            = 250
        self.window_length = 3
        self.min_samples   = self.fs

        # State
        self.current_state  = "Neutral"
        self.state_strength = 0.0
        self.state_history  = deque(maxlen=5)

        # Band powers (normalised)
        self.alpha_power = 0.0
        self.beta_power  = 0.0
        self.theta_power = 0.0
        self.gamma_power = 0.0

        # Per-channel alpha for FAA
        self.fp1_alpha = 0.0
        self.fp2_alpha = 0.0
        self.faa_raw   = 0.0   # ln(FP2α) − ln(FP1α)

        # Calibration
        self.faa_baseline   = None    # None = uncalibrated
        self.calibrating    = False
        self.calib_start    = 0.0
        self._calib_buf     = []
        self.calib_label    = "Press C to calibrate baseline"

        # Stress
        self.faa_stress  = 0.5
        self._stress_ema = 0.5
        self._ema_a      = 0.12

        # Session
        self.session_start = time.time()
        self.focus_time    = 0.0
        self.relax_time    = 0.0
        self._last_focus_t = time.time()
        self._last_relax_t = time.time()

        # Sparklines (~8 s)
        self.alpha_trend  = deque(maxlen=200)
        self.beta_trend   = deque(maxlen=200)
        self.stress_trend = deque(maxlen=200)

        # 60-second timeline: one entry per second
        self.timeline      = deque(maxlen=60)
        self._last_tl_time = 0.0

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _bandpass(self):
        nyq = 0.5 * self.fs
        return butter(3, [1 / nyq, 40 / nyq], btype='band')

    def _notch(self):
        nyq = 0.5 * self.fs
        return butter(2, [49 / nyq, 51 / nyq], btype='bandstop')

    def _welch(self, eeg):
        try:
            if len(eeg) < self.fs:
                return None, None
            nperseg = min(self.fs * 2, len(eeg) // 4)
            f, psd  = welch(eeg, self.fs, nperseg=nperseg, noverlap=nperseg // 2)
            v = np.isfinite(psd)
            return f[v], psd[v]
        except Exception:
            return None, None

    def _bp(self, f, psd, lo, hi):
        idx = (f >= lo) & (f <= hi)
        return float(np.trapezoid(psd[idx], f[idx])) if idx.sum() > 0 else 0.0

    # ------------------------------------------------------------------
    # Processing thread
    # ------------------------------------------------------------------

    def processing_thread(self):
        print("Dashboard: searching for ORIC stream…")
        streams = resolve_streams(wait_time=5.0)
        stream  = next((s for s in streams if s.name() == 'ORIC'), None)
        if stream is None:
            print("Dashboard: no ORIC stream found.")
            self.connection_status = "No stream"
            return

        inlet = StreamInlet(stream, max_chunklen=1)
        print("Dashboard: connected.")
        self.connection_status = "Connected"

        buf    = deque(maxlen=self.window_length * self.fs)
        bp_b, bp_a = self._bandpass()
        nt_b, nt_a = self._notch()
        eps = 1e-10

        while self.running:
            try:
                sample, _ = inlet.pull_sample(timeout=1.0)
                if sample is None:
                    continue

                buf.append(sample[:2])
                if len(buf) < self.min_samples:
                    continue

                data = np.array(buf)

                # Filter
                try:
                    df = filtfilt(bp_b, bp_a, data, axis=0)
                    df = filtfilt(nt_b, nt_a, df,   axis=0)
                except Exception:
                    df = data

                # Per-channel PSD → normalised band powers
                ch_res = []
                for ch in range(2):
                    f, psd = self._welch(df[:, ch])
                    if f is None:
                        continue
                    θ = self._bp(f, psd, 4,  8)
                    α = self._bp(f, psd, 8,  12)
                    β = self._bp(f, psd, 13, 30)
                    γ = self._bp(f, psd, 30, 40)
                    tot = θ + α + β + γ
                    if tot > 0:
                        ch_res.append({'θ': θ/tot, 'α': α/tot,
                                       'β': β/tot, 'γ': γ/tot})

                if len(ch_res) < 2:
                    continue

                self.fp1_alpha   = ch_res[0]['α']
                self.fp2_alpha   = ch_res[1]['α']
                self.alpha_power = (self.fp1_alpha + self.fp2_alpha) / 2
                self.beta_power  = (ch_res[0]['β'] + ch_res[1]['β']) / 2
                self.theta_power = (ch_res[0]['θ'] + ch_res[1]['θ']) / 2
                self.gamma_power = (ch_res[0]['γ'] + ch_res[1]['γ']) / 2

                # FAA
                faa = np.log(self.fp2_alpha + eps) - np.log(self.fp1_alpha + eps)
                self.faa_raw = float(faa)

                # Calibration collection
                if self.calibrating:
                    elapsed = time.time() - self.calib_start
                    if elapsed < self.CALIB_DURATION:
                        self._calib_buf.append(self.faa_raw)
                        remaining = self.CALIB_DURATION - elapsed
                        self.calib_label = f"Calibrating… {remaining:.0f}s — stay neutral"
                    else:
                        self.faa_baseline = float(np.mean(self._calib_buf)) \
                                            if self._calib_buf else 0.0
                        self.calibrating  = False
                        self.calib_label  = (f"Baseline set  "
                                             f"FAA={self.faa_baseline:+.3f}  |  C to redo")

                # Stress score
                if self.faa_baseline is not None:
                    # Deviation from personal baseline;
                    # negative deviation (more stressed than baseline) → high stress
                    dev = self.faa_baseline - self.faa_raw
                    raw_stress = float(np.clip(0.5 + dev / 1.5, 0.0, 1.0))
                else:
                    raw_stress = float(np.clip((2.0 - faa) / 4.0, 0.0, 1.0))

                self._stress_ema = (self._ema_a * raw_stress
                                    + (1 - self._ema_a) * self._stress_ema)
                self.faa_stress  = self._stress_ema

                # Trends
                self.alpha_trend.append(self.alpha_power)
                self.beta_trend.append(self.beta_power)
                self.stress_trend.append(self.faa_stress)

                # State classification
                α_avg, β_avg = self.alpha_power, self.beta_power
                total = α_avg + β_avg
                if total < 0.05:
                    state, strength = "Neutral", 0.1
                elif β_avg > α_avg * 1.1 and β_avg > 0.3:
                    state  = "Focus"
                    strength = min(β_avg / (total + eps), 1.0)
                elif α_avg > β_avg * 1.4 and α_avg > 0.3:
                    state  = "Relax"
                    strength = min(α_avg / (total + eps), 1.0)
                else:
                    state, strength = "Neutral", 0.3

                self.state_history.append((state, strength))
                if len(self.state_history) >= 5:
                    fv = sum(1 for s, _ in self.state_history if s == "Focus")
                    rv = sum(1 for s, _ in self.state_history if s == "Relax")
                    now = time.time()
                    if fv >= 3:
                        self.current_state  = "Focus"
                        self.state_strength = float(np.mean(
                            [s for st, s in self.state_history if st == "Focus"]))
                        self.focus_time += now - self._last_focus_t
                        self._last_focus_t = now
                    elif rv >= 3:
                        self.current_state  = "Relax"
                        self.state_strength = float(np.mean(
                            [s for st, s in self.state_history if st == "Relax"]))
                        self.relax_time += now - self._last_relax_t
                        self._last_relax_t = now
                    else:
                        self.current_state  = "Neutral"
                        self.state_strength = 0.3

                # Timeline — one entry per second
                now = time.time()
                if now - self._last_tl_time >= 1.0:
                    self.timeline.append({
                        'state':    self.current_state,
                        'stress':   self.faa_stress,
                        'strength': self.state_strength,
                    })
                    self._last_tl_time = now

            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _panel(self, x, y, w, h, label=None, label_right=None):
        import pygame
        pygame.draw.rect(screen, PANEL,  (x, y, w, h), border_radius=10)
        pygame.draw.rect(screen, BORDER, (x, y, w, h), 1, border_radius=10)
        if label:
            screen.blit(FONT.render(label, True, SUBTEXT), (x + 18, y + 14))
        if label_right:
            rw = FONT.size(label_right)[0]
            screen.blit(FONT.render(label_right, True, SUBTEXT),
                        (x + w - 24 - rw, y + 18))

    def _bar(self, x, y, w, h, value, color):
        import pygame
        pygame.draw.rect(screen, TRACK,  (x, y, w, h), border_radius=h // 2)
        fill = int(w * min(max(value, 0.0), 1.0))
        if fill > 0:
            pygame.draw.rect(screen, color, (x, y, fill, h), border_radius=h // 2)

    def _sparkline(self, data, x, y, w, h, color):
        import pygame
        pts = list(data)
        if len(pts) < 2:
            return
        mn, mx = min(pts), max(pts)
        rng = (mx - mn) or 1e-6
        coords = [
            (x + int(i / (len(pts) - 1) * w),
             y + h - int((v - mn) / rng * h))
            for i, v in enumerate(pts)
        ]
        pygame.draw.lines(screen, color, False, coords, 2)

    def _gauge(self, x, y, w, h, value,
               left_label, right_label,
               left_color, right_color,
               title=None, value_label=None):
        """
        Bipolar horizontal gauge.
        value: -1 (full left) to +1 (full right).
        """
        import pygame

        inner_y = y
        if title:
            tw = SMALL.size(title)[0]
            screen.blit(SMALL.render(title, True, SUBTEXT),
                        (x + w // 2 - tw // 2, inner_y))
            inner_y += SMALL.get_height() + 6

        track_h = max(10, h // 8)
        cy = inner_y + (y + h - inner_y) // 2
        cx = x + w // 2

        val = max(-1.0, min(1.0, float(value)))
        color = right_color if val >= 0 else left_color

        # Track
        pygame.draw.rect(screen, TRACK,
                         (x, cy - track_h // 2, w, track_h),
                         border_radius=track_h // 2)

        # Fill from center
        fill_w = int(abs(val) * w // 2)
        if fill_w > 0:
            if val > 0:
                pygame.draw.rect(screen, color,
                                 (cx, cy - track_h // 2, fill_w, track_h),
                                 border_radius=track_h // 2)
            else:
                pygame.draw.rect(screen, color,
                                 (cx - fill_w, cy - track_h // 2, fill_w, track_h),
                                 border_radius=track_h // 2)

        # Center tick
        pygame.draw.rect(screen, SUBTEXT, (cx - 1, cy - track_h, 2, track_h * 2))

        # Needle
        r  = max(12, track_h + 5)
        nx = int(cx + val * (w // 2))
        nx = max(x + r, min(x + w - r, nx))
        pygame.draw.circle(screen, PANEL,  (nx, cy), r)
        pygame.draw.circle(screen, color,  (nx, cy), r - 2)
        pygame.draw.circle(screen, PANEL,  (nx, cy), r // 3)

        # Value label above needle
        if value_label:
            vw = FONT.size(value_label)[0]
            screen.blit(FONT.render(value_label, True, color),
                        (nx - vw // 2, cy - r - FONT.get_height() - 4))

        # End labels — 10px up from bottom edge of gauge area
        screen.blit(FONT.render(left_label,  True, left_color),  (x, y + h - FONT.get_height() - 10))
        rw = FONT.size(right_label)[0]
        screen.blit(FONT.render(right_label, True, right_color), (x + w - rw, y + h - FONT.get_height() - 10))

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------

    def draw(self):
        import pygame
        screen.fill(BG)

        PAD = max(14, int(H * 0.018))

        # Fixed row heights — these sum to ~90% of H leaving natural breathing room
        HDR_H = int(H * 0.072)   # ~65px  header
        R1_H  = int(H * 0.28)    # ~252px gauges + state
        R2_H  = int(H * 0.14)    # ~126px FP1/FP2 alpha (extra top gap)
        R3_H  = int(H * 0.27)    # ~243px timeline
        R4_H  = int(H * 0.09)    # ~81px  stats bar (fixed, NOT residual)

        # ── Header ──────────────────────────────────────────────────────
        screen.blit(TITLE.render("EEG Dashboard", True, TEXT), (PAD, PAD))

        sess = time.time() - self.session_start
        screen.blit(SMALL.render(f"Session  {sess:.0f}s", True, SUBTEXT),
                    (PAD, PAD + TITLE.get_height() + 4))

        # Calibration status (centre of header)
        calib_color = STRESS_M if self.calibrating else (RELAX_C if self.faa_baseline is not None else SUBTEXT)
        cw = FONT.size(self.calib_label)[0]
        screen.blit(FONT.render(self.calib_label, True, calib_color),
                    (W // 2 - cw // 2, PAD + 6))

        # Connection status (top right)
        conn_c = RELAX_C if self.connection_status == "Connected" else STRESS_H
        conn_str = self.connection_status
        conn_w = FONT.size(conn_str)[0]
        pygame.draw.circle(screen, conn_c, (W - PAD - conn_w - 14, PAD + 13), 6)
        screen.blit(FONT.render(conn_str, True, conn_c), (W - PAD - conn_w, PAD + 6))

        content_y = PAD + HDR_H

        # ── Row 1: α/β gauge | State | FAA gauge ────────────────────────
        third = (W - PAD * 4) // 3
        gpad  = 28   # inner horizontal padding inside gauge panels
        top_pad = 46  # space from panel top to gauge interior (below panel label)

        stress_c = (STRESS_H if self.faa_stress > 0.65
                    else STRESS_M if self.faa_stress > 0.4
                    else STRESS_L)

        # α/β balance gauge
        gx = PAD
        self._panel(gx, content_y, third, R1_H, "Focus  ←  α/β balance  →  Relax")
        ab_raw = ((self.beta_power - self.alpha_power)
                  / (self.beta_power + self.alpha_power + 1e-10))
        ab_val = float(np.clip(ab_raw * 5.0, -1.0, 1.0))
        # Value readout in top-right corner of panel
        ab_readout = f"β {self.beta_power:.3f}  α {self.alpha_power:.3f}"
        rw = TINY.size(ab_readout)[0]
        screen.blit(TINY.render(ab_readout, True, SUBTEXT), (gx + third - 18 - rw, content_y + 16))
        self._gauge(
            gx + gpad, content_y + top_pad, third - gpad * 2, R1_H - top_pad - 12,
            ab_val,
            left_label="Relax", right_label="Focus",
            left_color=RELAX_C, right_color=FOCUS_C,
        )

        # State panel (centre)
        sx = PAD * 2 + third
        self._panel(sx, content_y, third, R1_H)
        state_c = {"Focus": FOCUS_C, "Relax": RELAX_C, "Neutral": NEUTRAL_C}[self.current_state]

        bl = BIG.render(self.current_state, True, state_c)
        screen.blit(bl, (sx + third // 2 - bl.get_width() // 2,
                          content_y + R1_H // 2 - bl.get_height() // 2 - 8))

        bar_y = content_y + R1_H - 44
        pct_str = f"{self.state_strength:.0%}"
        screen.blit(TINY.render("Strength", True, SUBTEXT), (sx + 20, bar_y - 20))
        screen.blit(TINY.render(pct_str, True, SUBTEXT),
                    (sx + third - 20 - TINY.size(pct_str)[0], bar_y - 20))
        self._bar(sx + 20, bar_y, third - 40, 10, self.state_strength, state_c)

        # FAA / Stress gauge (right)
        fx = PAD * 3 + third * 2
        calib_note = "personal baseline" if self.faa_baseline is not None else "population avg"
        self._panel(fx, content_y, third, R1_H, f"Stress (FAA)  —  {calib_note}")
        faa_readout = f"FAA {self.faa_raw:+.2f}  stress {self.faa_stress:.2f}"
        rw = TINY.size(faa_readout)[0]
        screen.blit(TINY.render(faa_readout, True, stress_c), (fx + third - 18 - rw, content_y + 16))

        if self.faa_baseline is not None:
            faa_val = float(np.clip((self.faa_raw - self.faa_baseline) / 1.5, -1.0, 1.0))
        else:
            faa_val = float(np.clip(self.faa_raw / 1.5, -1.0, 1.0))
        self._gauge(
            fx + gpad, content_y + top_pad, third - gpad * 2, R1_H - top_pad - 12,
            faa_val,
            left_label="Stress", right_label="Calm",
            left_color=STRESS_H, right_color=STRESS_L,
        )

        content_y += R1_H + PAD

        # ── Row 2: FP1 / FP2 alpha comparison ───────────────────────────
        self._panel(PAD, content_y, W - PAD * 2, R2_H,
                    "Frontal Alpha  —  FP1 left hemisphere  vs  FP2 right hemisphere")

        # Measure label/value widths to compute safe bar width
        lbl_w  = FONT.size("FP1")[0]
        val_w  = SMALL.size("0.000")[0]
        gap    = 18                          # space between label↔bar and bar↔value
        inner  = W - PAD * 2 - 36           # total inner width (18px each side)
        half   = inner // 2                  # each column
        bw     = half - lbl_w - val_w - gap * 3  # bar width per column

        row_y  = content_y + 62
        # col 1: FP1
        c1x = PAD + 18
        screen.blit(FONT.render("FP1", True, ALPHA_C), (c1x, row_y))
        self._bar(c1x + lbl_w + gap, row_y + 3, bw, 14, self.fp1_alpha * 2.5, ALPHA_C)
        screen.blit(SMALL.render(f"{self.fp1_alpha:.3f}", True, SUBTEXT),
                    (c1x + lbl_w + gap + bw + gap, row_y + 3))

        # col 2: FP2
        c2x = PAD + 18 + half + gap
        screen.blit(FONT.render("FP2", True, FOCUS_C), (c2x, row_y))
        self._bar(c2x + lbl_w + gap, row_y + 3, bw, 14, self.fp2_alpha * 2.5, FOCUS_C)
        screen.blit(SMALL.render(f"{self.fp2_alpha:.3f}", True, SUBTEXT),
                    (c2x + lbl_w + gap + bw + gap, row_y + 3))

        # Asymmetry note
        asym = self.fp1_alpha - self.fp2_alpha
        if asym > 0.03:
            asym_str = f"Δ = {asym:+.3f}   FP1 > FP2 → left resting → stress"
        elif asym < -0.03:
            asym_str = f"Δ = {asym:+.3f}   FP2 > FP1 → right resting → calm"
        else:
            asym_str = f"Δ = {asym:+.3f}   balanced"
        screen.blit(TINY.render(asym_str, True, SUBTEXT), (PAD + 18, row_y + 28))

        content_y += R2_H + PAD

        # ── Row 3: 60-second state + stress timeline ─────────────────────
        self._panel(PAD, content_y, W - PAD * 2, R3_H, "60-second history")

        tl_data = list(self.timeline)
        tl_x = PAD + 18
        tl_w = W - PAD * 2 - 36
        tl_y = content_y + 62

        if tl_data:
            state_colors  = {'Focus': FOCUS_C, 'Relax': RELAX_C, 'Neutral': NEUTRAL_C}
            stress_strip_h = 12
            bar_h  = R3_H - 92 - stress_strip_h

            cell_w = tl_w / 60
            for i, entry in enumerate(tl_data):
                cx  = int(tl_x + i * cell_w)
                cw  = max(1, int((i + 1) * cell_w) - int(i * cell_w))
                c   = state_colors.get(entry['state'], NEUTRAL_C)
                pygame.draw.rect(screen, c, (cx, tl_y, cw - 1, bar_h), border_radius=2)
                sh = int(entry['stress'] * stress_strip_h)
                if sh > 0:
                    pygame.draw.rect(screen, STRESS_H,
                                     (cx, tl_y + bar_h + 2, cw - 1, sh), border_radius=1)

            leg_y = tl_y + bar_h + stress_strip_h + 10
            legend = [("Focus", FOCUS_C), ("Relax", RELAX_C),
                      ("Neutral", NEUTRAL_C), ("Stress intensity", STRESS_H)]
            lx = tl_x
            for lbl, lc in legend:
                pygame.draw.rect(screen, lc, (lx, leg_y + 1, 10, 10), border_radius=2)
                screen.blit(TINY.render(lbl, True, SUBTEXT), (lx + 16, leg_y))
                lx += 10 + 16 + TINY.size(lbl)[0] + 28

            screen.blit(TINY.render("60s ago", True, SUBTEXT), (tl_x, tl_y - 20))
            nw = TINY.size("now")[0]
            screen.blit(TINY.render("now", True, SUBTEXT), (tl_x + tl_w - nw, tl_y - 20))
        else:
            msg = "Waiting for data…"
            mw = SMALL.size(msg)[0]
            screen.blit(SMALL.render(msg, True, SUBTEXT),
                        (PAD + (W - PAD * 2) // 2 - mw // 2,
                         content_y + R3_H // 2 - 8))

        content_y += R3_H + PAD

        # ── Row 4: Stats bar (fixed height) ─────────────────────────────
        self._panel(PAD, content_y, W - PAD * 2, R4_H)
        stats = [
            ("α",      f"{self.alpha_power:.3f}", ALPHA_C),
            ("β",      f"{self.beta_power:.3f}",  BETA_C),
            ("θ",      f"{self.theta_power:.3f}", NEUTRAL_C),
            ("Focus",  f"{self.focus_time:.0f}s", FOCUS_C),
            ("Relax",  f"{self.relax_time:.0f}s", RELAX_C),
            ("Stress", f"{self.faa_stress:.2f}",  stress_c),
        ]
        n     = len(stats)
        col_w = (W - PAD * 2) // n
        by    = content_y + (R4_H - TINY.get_height() - FONT.get_height() - 6) // 2
        for i, (label, val, col) in enumerate(stats):
            bx = PAD + 22 + i * col_w
            screen.blit(TINY.render(label, True, SUBTEXT), (bx, by))
            screen.blit(FONT.render(val, True, col), (bx, by + TINY.get_height() + 6))

        # Calibration progress bar at very bottom
        if self.calibrating:
            remaining = self.CALIB_DURATION - (time.time() - self.calib_start)
            prog = 1.0 - remaining / self.CALIB_DURATION
            bar_y2 = H - PAD - 6
            pygame.draw.rect(screen, TRACK, (PAD, bar_y2, W - PAD * 2, 6), border_radius=3)
            pygame.draw.rect(screen, STRESS_M,
                             (PAD, bar_y2, int((W - PAD * 2) * prog), 6), border_radius=3)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Headless debug loop
    # ------------------------------------------------------------------

    def _debug_loop(self, log_fh):
        hdr = (f"{'TIME':8}  {'STATE':7}  {'STR':4}  "
               f"{'α':6}  {'β':6}  {'θ':6}  "
               f"{'FP1α':6}  {'FP2α':6}  {'FAA':7}  {'STRESS':6}")
        print(hdr); print("-" * len(hdr))
        if log_fh:
            log_fh.write(hdr + "\n" + "-" * len(hdr) + "\n")
        while self.running:
            time.sleep(1.0)
            line = (
                f"{time.strftime('%H:%M:%S'):8}  {self.current_state:7}  "
                f"{self.state_strength:4.2f}  "
                f"{self.alpha_power:6.3f}  {self.beta_power:6.3f}  "
                f"{self.theta_power:6.3f}  "
                f"{self.fp1_alpha:6.3f}  {self.fp2_alpha:6.3f}  "
                f"{self.faa_raw:+7.3f}  {self.faa_stress:6.3f}"
            )
            print(line, flush=True)
            if log_fh:
                log_fh.write(line + "\n"); log_fh.flush()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, no_gui=False, log_file=None):
        log_fh = open(log_file, "w") if log_file else None
        t = threading.Thread(target=self.processing_thread, daemon=True)
        t.start()

        if no_gui:
            print("[dashboard] headless mode — Ctrl-C to stop\n")
            try:
                self._debug_loop(log_fh)
            except KeyboardInterrupt:
                pass
            finally:
                self.running = False
                if log_fh: log_fh.close()
                print("\nDashboard stopped.")
            return

        pg = _init_gui()
        clock = pg.time.Clock()
        try:
            while self.running:
                for ev in pg.event.get():
                    if ev.type == pg.QUIT:
                        self.running = False
                    elif ev.type == pg.KEYDOWN:
                        if ev.key in (pg.K_q, pg.K_ESCAPE):
                            self.running = False
                        elif ev.key == pg.K_c and not self.calibrating:
                            self.calibrating  = True
                            self.calib_start  = time.time()
                            self._calib_buf   = []
                            self.calib_label  = f"Calibrating… {self.CALIB_DURATION:.0f}s — stay neutral"
                self.draw()
                clock.tick(30)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            pg.quit()
            if log_fh: log_fh.close()
            print("Dashboard stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Dashboard")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless: print metrics to stdout")
    parser.add_argument("--log", metavar="FILE",
                        help="Write headless log to FILE")
    args = parser.parse_args()

    d = EEGDashboard()
    d.run(no_gui=args.no_gui, log_file=args.log)
