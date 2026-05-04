"""
Focus Training Game — real-time EEG neurofeedback loop.

Closed-loop: Brain State → Signal → Mapping → Game Response → Reinforcement → Brain State

Run:
    python focus_game.py              # 5-min mode (default)
    python focus_game.py --mode 2min  # 2-min onboarding mode
    python focus_game.py --no-lsl     # synthetic demo (no hardware)
"""

import argparse
import math
import random
import threading
import time
from collections import deque

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, iirnotch, welch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SRATE = 250                  # Hz — fixed by ORIC hardware
CHANNELS = 2                 # FP1 (0), FP2 (1)
BUFFER_SECS = 4              # rolling signal buffer length
BUFFER_LEN = BUFFER_SECS * SRATE

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (13, 30),
    "gamma": (30, 40),
}

EPS = 1e-8

# Scoring / physics
DT = 1 / 30                  # game loop tick (seconds)

# ─────────────────────────────────────────────────────────────────────────────
# Session mode presets
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "2min": {
        "duration": 120,
        "threshold_percentile": 60,
        "alpha_gain": 1.2,
        "exponent_p": 1.2,
        "k_stability": 0.6,
        "label": "2-Min Focus Sprint",
    },
    "5min": {
        "duration": 300,
        "threshold_percentile": 68,
        "alpha_gain": 0.9,
        "exponent_p": 1.8,
        "k_stability": 0.4,
        "label": "5-Min Deep Training",
    },
}

# Phase boundaries (fraction of session duration)
PHASE_1_END = 30          # seconds — fixed calibration window
PHASE_3_START = 0.70      # fraction of remaining session

# ─────────────────────────────────────────────────────────────────────────────
# Signal processing helpers  (mirrors controller.py, kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _butter_bandpass(low, high, fs, order=3):
    nyq = fs / 2
    return butter(order, [low / nyq, high / nyq], btype="band")


def _notch(freq, fs, Q=30):
    return iirnotch(freq / (fs / 2), Q)


def _apply_filter(b, a, data):
    if len(data) < max(len(b), len(a)) * 3:
        return data
    try:
        return filtfilt(b, a, data)
    except Exception:
        return data


def _bandpower(freqs, psd, low, high):
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    if idx.size > 1:
        try:
            return float(np.trapezoid(psd[idx], freqs[idx]))
        except AttributeError:
            return float(np.trapz(psd[idx], freqs[idx]))
    return 0.0


def _robust_welch(signal, fs):
    nperseg = min(fs * 2, len(signal))
    if nperseg < 16:
        return np.array([0.0]), np.array([0.0])
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


# Pre-build filters once
_BP = {name: _butter_bandpass(lo, hi, SRATE) for name, (lo, hi) in BANDS.items()}
_NOTCH_B, _NOTCH_A = _notch(50, SRATE)
_BP1_40_B, _BP1_40_A = butter(3, [1 / 125, 40 / 125], btype="band")


def extract_band_powers(raw_ch: np.ndarray) -> dict:
    """Return normalized band powers for one channel array (N samples)."""
    sig = _apply_filter(_BP1_40_B, _BP1_40_A, raw_ch)
    sig = _apply_filter(_NOTCH_B, _NOTCH_A, sig)
    freqs, psd = _robust_welch(sig, SRATE)
    powers = {name: _bandpower(freqs, psd, lo, hi) for name, (lo, hi) in BANDS.items()}
    total = sum(powers.values()) + EPS
    return {k: v / total for k, v in powers.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Focus feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class FocusProcessor:
    """
    Converts raw FP1/FP2 samples → composite focus score F_total in [0, 1].

    Pipeline:
        raw EEG → band powers → engagement index F
                              → stability component
                              → F_total = 0.7·F + 0.3·stability_norm
    """

    # F history for stability (3 sec at ~30 fps tick rate)
    F_HISTORY_LEN = 90
    STABILITY_K   = 200.0   # tuned for var(F∈[0,1]): var≈0.005→stable, var≈0.05→jittery

    # EMA smoothing on F before exposing as Current Strength
    F_EMA_ALPHA = 0.15

    def __init__(self):
        self.buffer    = deque(maxlen=BUFFER_LEN)   # (fp1, fp2) tuples
        self.F_history = deque(maxlen=self.F_HISTORY_LEN)  # for stability variance
        self._F_ema    = 0.3                                # smoothed current strength

    def push(self, fp1: float, fp2: float):
        self.buffer.append((fp1, fp2))

    def compute(self) -> dict | None:
        if len(self.buffer) < SRATE:
            return None

        data = np.array(self.buffer, dtype=np.float32)  # (N, 2)
        fp1_raw, fp2_raw = data[:, 0], data[:, 1]

        bp1 = extract_band_powers(fp1_raw)
        bp2 = extract_band_powers(fp2_raw)

        alpha = (bp1["alpha"] + bp2["alpha"]) / 2
        beta  = (bp1["beta"]  + bp2["beta"])  / 2
        theta = (bp1["theta"] + bp2["theta"]) / 2

        # Raw focus: engagement index β/(α+θ)
        F_raw = beta / (alpha + theta + EPS)
        F_raw = float(np.clip(F_raw, 0, 1))

        # EMA smoothing — spec: S_t = α·P_t + (1-α)·S_{t-1}
        self._F_ema = self.F_EMA_ALPHA * F_raw + (1 - self.F_EMA_ALPHA) * self._F_ema
        F = float(self._F_ema)

        # Stability — spec: σ² = var(S over 3s), Stability = 1 - k·σ²
        # Use var(F history) — focus signal variance, NOT raw EEG amplitude
        self.F_history.append(F)
        if len(self.F_history) >= 5:
            variance = float(np.var(np.array(self.F_history)))
        else:
            variance = 0.05   # assume jittery until enough history
        stability = 1.0 / (variance + EPS)
        stability_norm = float(np.clip(stability / self.STABILITY_K, 0, 1))

        # stability_norm is used only for scoring/flow — not baked into F_total
        # (including it here creates a constant upward bias that moves the ball
        #  even during neutral since stable signals always have high stability_norm)
        F_total = F

        # FAA stress (for display only)
        faa = math.log(bp2["alpha"] + EPS) - math.log(bp1["alpha"] + EPS)
        faa_norm = float(np.clip((faa + 2) / 4, 0, 1))

        return {
            "F": F,
            "stability": stability_norm,
            "F_total": F_total,
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "faa_stress": faa_norm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Game engine (scoring, normalization, flow, difficulty)
# ─────────────────────────────────────────────────────────────────────────────

class FocusGameEngine:
    """
    Stateful game engine that consumes F_total samples and produces:
        - normalized control signal Z
        - mapped output y
        - physics: velocity, position, charge
        - score components
        - flow state
        - adaptive threshold / gain
    """

    # EMA coefficients
    BETA_FAST = 0.05      # fast baseline (≈20 sample half-life)
    BETA_SLOW = 0.005     # slow baseline (≈200 sample half-life)

    # MAD window for robust variance
    MAD_WINDOW = 60       # samples of F_total history

    # Momentum / physics
    GAMMA = 0.85          # velocity smoothing
    CHARGE_DECAY = 0.995  # per-tick charge decay

    # Scoring
    RECOVERY_BONUS = 15.0
    SPIKE_PENALTY = 0.9
    SPIKE_THRESHOLD = 0.35       # absolute jump in F_total in one tick

    # Flow — spec requires BOTH strength AND stability above threshold
    FLOW_ONSET_SEC      = 1.0    # must be above θ this long to enter flow
    FLOW_STABILITY_MIN  = 0.25   # minimum stability_norm to count toward flow

    # Difficulty adaptation
    PERCENTILE_CLAMP = (60, 80)
    ALPHA_GAIN_CLAMP = (0.5, 2.5)
    ALPHA_GAIN_STEP = 0.02

    def __init__(self, mode: str):
        cfg = MODES[mode]
        self.duration = cfg["duration"]
        self.target_percentile = float(cfg["threshold_percentile"])
        self.alpha_gain = cfg["alpha_gain"]
        self.exponent_p = cfg["exponent_p"]
        self.k_stability = cfg["k_stability"]

        # Normalization state
        self.mu_short = 0.5
        self.mu_long  = 0.5
        self.F_history = deque(maxlen=self.MAD_WINDOW)

        # Adaptive threshold
        self.strength_30s = deque(maxlen=int(30 / DT))
        self.theta = 0.5         # will be updated once buffer fills
        self._above_theta_count = 0
        self._total_count = 0

        # Physics
        self.velocity = 0.0
        self.position = 0.5      # [0, 1]; 0.5 = center
        self.charge = 0.0

        # Scoring
        self.score = 0.0
        self.prev_strength = 0.0
        self.prev_F_total = 0.0

        # True velocity: smoothed dS/dt  (spec: V_t = (S_t - S_{t-Δt}) / Δt)
        self.strength_velocity: float = 0.0
        self._prev_strength_raw: float = 0.0

        # Flow
        self.flow_active = False
        self.flow_timer = 0.0        # seconds continuously above θ
        self.flow_time_total = 0.0   # cumulative flow seconds this session
        self.longest_flow = 0.0
        self.flow_entries = 0
        self.flow_strength_acc = []  # for quality metric

        # Session timing / phases
        self.elapsed = 0.0
        self.phase = 1               # 1, 2, or 3

        # Output state (thread-safe read)
        self.Z = 0.0
        self.y = 0.0
        self.strength = 0.0
        self.stability_norm = 0.0
        self.F_total = 0.0

    # ── normalization ─────────────────────────────────────────────────────────

    def _update_baseline(self, F_total: float):
        b = self.BETA_FAST
        self.mu_short = (1 - b) * self.mu_short + b * F_total
        b = self.BETA_SLOW
        self.mu_long  = (1 - b) * self.mu_long  + b * F_total
        self.F_history.append(F_total)

    def _normalize(self, F_total: float) -> float:
        mu = 0.7 * self.mu_short + 0.3 * self.mu_long
        if len(self.F_history) >= 5:
            arr = np.array(self.F_history)
            sigma = float(np.median(np.abs(arr - np.median(arr))))
        else:
            sigma = 0.1
        Z = (F_total - mu) / (sigma + EPS)
        return float(np.clip(Z, -3, 3))

    # ── control mapping ───────────────────────────────────────────────────────

    def _map(self, Z: float) -> float:
        if abs(Z) < 0.2:
            return 0.0
        return math.tanh(self.alpha_gain * Z)

    # ── difficulty adaptation ─────────────────────────────────────────────────

    def _update_difficulty(self):
        if len(self.strength_30s) < 10:
            return
        arr = np.array(self.strength_30s)
        self.theta = float(np.percentile(arr, self.target_percentile))

        # Adapt percentile target based on success rate
        if self._total_count > 0:
            sr = self._above_theta_count / self._total_count
            if sr > 0.70:
                self.target_percentile = min(self.target_percentile + 2,
                                             self.PERCENTILE_CLAMP[1])
            else:
                self.target_percentile = max(self.target_percentile - 2,
                                             self.PERCENTILE_CLAMP[0])
            self._above_theta_count = 0
            self._total_count = 0

        # Adapt gain: if Z variance is low → increase gain (push sensitivity)
        if len(self.F_history) >= self.MAD_WINDOW:
            Z_var = float(np.var(np.array(self.F_history)))
            if Z_var < 0.01:
                self.alpha_gain = min(self.alpha_gain + self.ALPHA_GAIN_STEP,
                                      self.ALPHA_GAIN_CLAMP[1])
            elif Z_var > 0.05:
                self.alpha_gain = max(self.alpha_gain - self.ALPHA_GAIN_STEP,
                                      self.ALPHA_GAIN_CLAMP[0])

    # ── phase management ──────────────────────────────────────────────────────

    def _current_phase(self) -> int:
        if self.elapsed < PHASE_1_END:
            return 1
        remaining = self.duration - self.elapsed
        phase3_start = self.duration * (1 - PHASE_3_START)
        if remaining < phase3_start:
            return 3
        return 2

    def _phase_theta_modifier(self) -> float:
        """Lower threshold in phase 1, raise in phase 3."""
        if self.phase == 1:
            return -0.05
        if self.phase == 3:
            return 0.05
        return 0.0

    def _phase_score_bonus(self) -> float:
        return 1.5 if self.phase == 3 else 1.0

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self, features: dict) -> dict:
        """
        Consume one FocusProcessor output dict.
        Returns engine state dict for the renderer.
        """
        F_total = features["F_total"]
        stab    = features["stability"]
        dt = DT

        self.elapsed += dt
        self.phase = self._current_phase()

        # Spike detection (before baseline update)
        sudden_jump = abs(F_total - self.prev_F_total) > self.SPIKE_THRESHOLD
        self.prev_F_total = F_total

        # Normalization
        self._update_baseline(F_total)
        Z = self._normalize(F_total)
        y = self._map(Z)

        # Physics — weak spring pulls position back toward centre (0.5)
        # prevents ball from coasting indefinitely when signal returns to neutral
        spring = -0.3 * (self.position - 0.5)
        self.velocity = self.GAMMA * self.velocity + (1 - self.GAMMA) * (y + spring)
        self.position = self.position + self.velocity * dt   # unbounded

        # Charge (effort accumulation)
        self.charge += max(0.0, y) * dt
        self.charge *= self.CHARGE_DECAY

        # Adaptive threshold
        strength = float(np.clip(F_total, 0, 1))
        self.strength_30s.append(strength)
        self._total_count += 1
        effective_theta = self.theta + self._phase_theta_modifier()
        if strength > effective_theta:
            self._above_theta_count += 1
        if len(self.strength_30s) == self.strength_30s.maxlen:
            self._update_difficulty()

        # Velocity: smoothed first derivative of strength (spec: V_t = dS/dt)
        raw_vel = (strength - self._prev_strength_raw) / dt
        self.strength_velocity = 0.7 * self.strength_velocity + 0.3 * raw_vel
        self._prev_strength_raw = strength

        # ── Flow state — spec: strength > T_strength AND stability > T_stability ──
        in_flow_zone = strength > effective_theta and stab >= self.FLOW_STABILITY_MIN
        if in_flow_zone:
            self.flow_timer += dt
            self.flow_strength_acc.append(strength - effective_theta)
        else:
            if self.flow_active:
                self.longest_flow = max(self.longest_flow, self.flow_timer)
                self.flow_active = False
                self.flow_timer = 0.0
                self.flow_strength_acc = []
            else:
                self.flow_timer = max(0.0, self.flow_timer - dt * 0.5)

        if not self.flow_active and self.flow_timer >= self.FLOW_ONSET_SEC:
            self.flow_active = True
            self.flow_entries += 1

        if self.flow_active:
            self.flow_time_total += dt

        flow_quality = (float(np.mean(self.flow_strength_acc))
                        if self.flow_strength_acc else 0.0)
        flow_multiplier = 1.0 + math.log(1.0 + self.flow_time_total) if self.flow_active else 1.0

        # ── Scoring ───────────────────────────────────────────────────────────
        delta_score = 0.0

        if strength > effective_theta:
            excess = strength - effective_theta
            delta_score += dt * (excess ** self.exponent_p)

        # stability bonus
        delta_score += dt * stab * self.k_stability

        # apply flow multiplier and phase bonus
        delta_score *= flow_multiplier * self._phase_score_bonus()

        # recovery bonus
        if self.prev_strength <= effective_theta and strength > effective_theta:
            delta_score += self.RECOVERY_BONUS

        # anti-spike penalty
        if sudden_jump:
            self.score *= self.SPIKE_PENALTY

        self.score += delta_score
        self.prev_strength = strength

        # Expose state
        self.Z = Z
        self.y = y
        self.strength = strength
        self.stability_norm = stab
        self.F_total = F_total

        return {
            "elapsed": self.elapsed,
            "phase": self.phase,
            "Z": Z,
            "y": y,
            "position": self.position,
            "velocity": self.velocity,
            "charge": self.charge,
            "strength": strength,
            "stability": stab,
            "theta": effective_theta,
            "flow_active": self.flow_active,
            "flow_timer": self.flow_timer,
            "flow_quality": flow_quality,
            "flow_time_total": self.flow_time_total,
            "longest_flow": self.longest_flow,
            "flow_entries": self.flow_entries,
            "score": self.score,
            "strength_velocity": self.strength_velocity,
            "alpha_gain": self.alpha_gain,
            "target_percentile": self.target_percentile,
            "faa_stress": features.get("faa_stress", 0.5),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LSL reader thread
# ─────────────────────────────────────────────────────────────────────────────

class EEGThread(threading.Thread):
    """Reads LSL stream, pushes samples into FocusProcessor, exposes latest features."""

    def __init__(self, processor: FocusProcessor):
        super().__init__(daemon=True)
        self.processor = processor
        self.latest: dict | None = None
        self.running = True
        self._lock = threading.Lock()

    def get_latest(self) -> dict | None:
        with self._lock:
            return self.latest

    def run(self):
        from pylsl import StreamInlet, resolve_streams
        print("Searching for ORIC EEG stream…")
        streams = resolve_streams(wait_time=5.0)
        oric = next((s for s in streams if s.name() == "ORIC"), None)
        if oric is None:
            print("No ORIC stream found. Run synth streamer or connect hardware.")
            return
        inlet = StreamInlet(oric, max_chunklen=1)
        print("Connected to ORIC stream.")
        while self.running:
            sample, _ = inlet.pull_sample(timeout=0.1)
            if sample is None:
                continue
            fp1, fp2 = float(sample[0]), float(sample[1])
            self.processor.push(fp1, fp2)
            features = self.processor.compute()
            if features is not None:
                with self._lock:
                    self.latest = features


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic EEG source (no-LSL mode for testing)
# ─────────────────────────────────────────────────────────────────────────────

class SynthThread(threading.Thread):
    """Generates synthetic EEG and pushes through FocusProcessor."""

    def __init__(self, processor: FocusProcessor):
        super().__init__(daemon=True)
        self.processor = processor
        self.latest: dict | None = None
        self.running = True
        self._lock = threading.Lock()
        self._t = 0.0

    def get_latest(self) -> dict | None:
        with self._lock:
            return self.latest

    def _synth_sample(self):
        t = self._t
        # Slowly oscillating focus state with noise
        focus_phase = math.sin(t / 8) * 0.5 + 0.5  # slow 0–1 envelope
        alpha_amp = 1.0 - focus_phase * 0.7
        beta_amp  = 0.3 + focus_phase * 0.7

        fp1 = (alpha_amp * math.sin(2 * math.pi * 10 * t)
               + beta_amp  * math.sin(2 * math.pi * 20 * t)
               + random.gauss(0, 0.1))
        fp2 = (alpha_amp * math.sin(2 * math.pi * 10 * t + 0.3)
               + beta_amp  * math.sin(2 * math.pi * 20 * t + 0.1)
               + random.gauss(0, 0.1))
        return fp1, fp2

    def run(self):
        interval = 1.0 / SRATE
        while self.running:
            fp1, fp2 = self._synth_sample()
            self._t += interval
            self.processor.push(fp1, fp2)
            features = self.processor.compute()
            if features is not None:
                with self._lock:
                    self.latest = features
            time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
# Renderer helpers
# ─────────────────────────────────────────────────────────────────────────────

# Purple → cyan gradient endpoints for bars
_BAR_C1 = np.array([123, 104, 238], dtype=np.float32)
_BAR_C2 = np.array([64,  224, 208], dtype=np.float32)


def _gradient_bar(surf, rect, frac, radius=3):
    """Draw a purple→cyan gradient progress bar."""
    pygame.draw.rect(surf, (22, 25, 42), rect, border_radius=radius)
    fw = int(rect.width * max(0.0, min(1.0, frac)))
    if fw <= 0:
        return
    for i in range(fw):
        t = i / max(fw - 1, 1)
        c = (_BAR_C1 + (_BAR_C2 - _BAR_C1) * t).astype(np.uint8)
        pygame.draw.line(surf, tuple(int(x) for x in c),
                         (rect.left + i, rect.top + 1),
                         (rect.left + i, rect.bottom - 1))


def _panel(surf, rect, radius=10):
    """Dark glassmorphism panel."""
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    s.fill((12, 15, 28, 210))
    pygame.draw.rect(s, (55, 65, 105, 90), s.get_rect(), 1, border_radius=radius)
    surf.blit(s, rect.topleft)


def _mono_fonts():
    """Return (xl, lg, md, sm) in a monospace face."""
    candidates = ["Menlo", "Consolas", "Courier New", "Courier", "monospace"]
    def get(size, bold=False):
        for name in candidates:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f:
                return f
        return pygame.font.SysFont(None, size, bold=bold)
    return get(64, True), get(22, False), get(17), get(13)


# ─────────────────────────────────────────────────────────────────────────────
# Main game class  — dark sci-fi orbital HUD
# ─────────────────────────────────────────────────────────────────────────────

class FocusGame:

    W, H = 1280, 720

    # Wave history: ~7 sec at 30 fps
    WAVE_LEN = 210

    # Ball is the leading edge of the wave at 65% from left
    BALL_X_FRAC = 0.65

    # Wave occupies vertical centre strip
    WAVE_TOP_PX = 130    # y-min for wave (highest it can go)
    WAVE_BOT_PX = 200    # margin from bottom  →  floor = H - 200 = 520

    # Colours
    C_BG    = (7,   9,  18)
    C_GRID  = (38, 46, 78)
    C_WHITE = (255, 255, 255)
    C_DIM   = (90, 100, 130)
    C_LABEL = (120, 135, 170)
    C_CYAN  = (64, 224, 208)
    C_PURP  = (123, 104, 238)
    C_FLOW_BADGE = (40, 210, 195)

    def __init__(self, mode: str, use_lsl: bool):
        self.mode = mode
        self.cfg = MODES[mode]
        self.processor = FocusProcessor()
        self.engine = FocusGameEngine(mode)

        if use_lsl:
            self.eeg_thread = EEGThread(self.processor)
        else:
            self.eeg_thread = SynthThread(self.processor)
        self.eeg_thread.start()

        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("FRONTAL NEUROTRAINER · ORBITAL SUITE")
        self.clock = pygame.time.Clock()

        self.f_xl, self.f_lg, self.f_md, self.f_sm = _mono_fonts()

        self.state = "waiting"
        self.engine_state: dict = {}

        self._wave_history: deque = deque([0.5] * self.WAVE_LEN, maxlen=self.WAVE_LEN)
        self._top_strength: int = 0
        self._fullscreen = False
        self._ball_angle: float = 0.0
        self._grid_surf: pygame.Surface | None = None
        self._paused = False
        self._pause_rect: pygame.Rect | None = None
        self._tick: int = 0          # incremented each drawn frame, used for sparks

    # ── grid (built once) ────────────────────────────────────────────────────

    def _ensure_grid(self):
        if self._grid_surf is not None:
            return
        s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for x in range(0, self.W, 60):
            pygame.draw.line(s, (*self.C_GRID, 35), (x, 0), (x, self.H))
        for y in range(0, self.H, 60):
            pygame.draw.line(s, (*self.C_GRID, 35), (0, y), (self.W, y))
        self._grid_surf = s

    # ── wave geometry ─────────────────────────────────────────────────────────

    def _wave_bounds(self):
        top = int(self.H * 0.18)
        bot = self.H - int(self.H * 0.28)
        return top, bot

    def _pos_to_y(self, pos: float) -> int:
        top, bot = self._wave_bounds()
        return int(bot - float(np.clip(pos, 0, 1)) * (bot - top))

    # Pixels per 1 unit of position change — tunes how "tall" the wave swings look
    WAVE_SCALE = 0.11    # fraction of H  →  ~79px/unit at 720p

    def _ball_screen_y(self) -> int:
        """Ball is always drawn at this fixed vertical centre on screen."""
        return int(self.H * 0.42)

    def _get_wave_ys(self, sigma: float = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Camera-relative trail: ball stays fixed at _ball_screen_y().
        Trail points are offset from ball by (current_pos - point_pos) * scale_px.
        sigma controls gaussian smoothing — lower = more data oscillation visible.
        """
        ball_x  = int(self.W * self.BALL_X_FRAC)
        ball_sy = self._ball_screen_y()
        scale   = int(self.H * self.WAVE_SCALE)

        hist = np.array(self._wave_history, dtype=np.float32)
        n = len(hist)

        if n < 2:
            wave_ys = np.full(self.W, self.H + 10, dtype=np.int32)
            wave_ys[:ball_x] = ball_sy
            return wave_ys, np.zeros(ball_x, dtype=np.float32)

        if n < self.WAVE_LEN:
            pad = np.full(self.WAVE_LEN - n, hist[0], dtype=np.float32)
            hist = np.concatenate([pad, hist])

        src = np.linspace(0, 1, self.WAVE_LEN, dtype=np.float32)
        dst = np.linspace(0, 1, ball_x, dtype=np.float32)
        trail = np.interp(dst, src, hist).astype(np.float32)
        trail_smooth = gaussian_filter1d(trail, sigma=sigma)

        current_pos = float(trail_smooth[-1])
        trail_ys = (ball_sy + (current_pos - trail_smooth) * scale).astype(np.int32)

        wave_ys = np.full(self.W, self.H + 10, dtype=np.int32)
        wave_ys[:ball_x] = trail_ys
        return wave_ys, trail_smooth

    # ── trail: glowing line fading old→new ───────────────────────────────────

    @staticmethod
    def _motion_mode(es: dict) -> str:
        """Return 'flow' | 'rising' | 'falling' | 'neutral'."""
        if es.get("flow_active", False):
            return "flow"
        vel = es.get("strength_velocity", 0.0)
        if vel > 0.05:
            return "rising"
        if vel < -0.05:
            return "falling"
        return "neutral"

    def _draw_trail(self, wave_ys: np.ndarray, ball_x: int, mode: str, es: dict):
        """
        Physics-aware trail renderer.
        flow    → rocket booster: thick cyan exhaust, widening near ball, sparks
        falling → falling comet: red/orange, widens at head (ball), tapers back
        rising  → slow climb: thin white gradient, quiet
        neutral → data wave: very thin, let EEG oscillation show through
        """
        step = 2
        pts = [(x, int(wave_ys[x])) for x in range(0, ball_x, step)]
        n = len(pts)
        if n < 2:
            return

        if mode == "flow":
            # ── ROCKET BOOSTER ─────────────────────────────────────────
            # Layer 1: full trail, thin dim cyan
            for i in range(1, n):
                t = i / max(n - 1, 1)
                c = (0, int(80 + 60 * t), int(80 + 80 * t))
                pygame.draw.line(self.screen, c, pts[i - 1], pts[i], 1)

            # Layer 2: last 55%, medium width, brighter cyan
            s2 = int(n * 0.45)
            for i in range(s2 + 1, n):
                t = (i - s2) / max(n - s2, 1)
                c = (0, int(140 + 84 * t), int(150 + 58 * t))
                pygame.draw.line(self.screen, c, pts[i - 1], pts[i], 3)

            # Layer 3: last 25%, thick bright white-cyan
            s3 = int(n * 0.75)
            for i in range(s3 + 1, n):
                t = (i - s3) / max(n - s3, 1)
                r_val = int(180 * t)
                c = (r_val, int(220 + 35 * t), 255)
                w = int(5 + 4 * t)   # 5→9px near ball
                pygame.draw.line(self.screen, c, pts[i - 1], pts[i], w)

            # Spark particles — 8 bright dots in last 18%, positions jittered per tick
            rng = random.Random(self._tick * 7 + 13)
            s4 = int(n * 0.82)
            for _ in range(8):
                idx = rng.randint(s4, n - 1)
                px, py = pts[idx]
                ox = rng.randint(-6, 6)
                oy = rng.randint(-6, 6)
                sr = rng.randint(1, 3)
                alpha = rng.randint(120, 230)
                sg = pygame.Surface((sr * 2 + 2, sr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(sg, (255, 255, 255, alpha), (sr + 1, sr + 1), sr)
                self.screen.blit(sg, (px + ox - sr - 1, py + oy - sr - 1))

        elif mode == "falling":
            # ── FALLING COMET ──────────────────────────────────────────
            # Trail widens toward ball (head of comet), tapers to past (tail)
            # Colours: dark orange (old) → orange → red → bright near ball
            for i in range(1, n):
                t = i / max(n - 1, 1)          # 0 = oldest, 1 = newest (ball)
                w = max(1, int(1 + 6 * (t ** 2)))  # 1px old → 7px near ball
                # dark orange → red → bright red-white
                r = min(255, int(120 + 135 * t))
                g = int(60 * (1 - t) * (1 - t))    # orange tint fades out
                b = 0
                pygame.draw.line(self.screen, (r, g, b), pts[i - 1], pts[i], w)

            # Glow halo over last 20%: red-orange smear
            s2 = int(n * 0.80)
            for i in range(s2 + 1, n):
                t2 = (i - s2) / max(n - s2, 1)
                alpha = int(30 + 40 * t2)
                gw = int(4 + 8 * t2)
                gs = pygame.Surface((gw * 2 + 2, gw * 2 + 2), pygame.SRCALPHA)
                pygame.draw.line(
                    gs, (255, int(80 * (1 - t2)), 0, alpha),
                    (gw + 1 + pts[i-1][0] - pts[i][0], gw + 1 + pts[i-1][1] - pts[i][1]),
                    (gw + 1, gw + 1), gw)
                self.screen.blit(gs, (pts[i][0] - gw - 1, pts[i][1] - gw - 1))

        elif mode == "rising":
            # ── SLOW CLIMB ────────────────────────────────────────────
            # Thin, clean white gradient — understated, signal is recovering
            for i in range(1, n):
                t = (i / max(n - 1, 1)) ** 1.6
                v = int(40 + 200 * t)
                pygame.draw.line(self.screen, (v, v, min(v + 20, 255)),
                                 pts[i - 1], pts[i], 1)

        else:
            # ── NEUTRAL DATA WAVE ──────────────────────────────────────
            # Very thin, muted — EEG oscillation itself creates the wave shape
            # (controlled by sigma=4 in _get_wave_ys → data bumps visible)
            for i in range(1, n):
                t = (i / max(n - 1, 1)) ** 1.2
                v = int(25 + 170 * t)
                pygame.draw.line(self.screen, (v, v, min(v + 15, 255)),
                                 pts[i - 1], pts[i], 1)

    # ── ball ──────────────────────────────────────────────────────────────────

    def _draw_ball(self, wave_ys: np.ndarray, es: dict, mode: str):
        ball_x = int(self.W * self.BALL_X_FRAC)
        ball_y = self._ball_screen_y()
        strength = es.get("strength", 0.0)

        # Spin speed varies by mode
        spin_rates = {"flow": 45.0, "falling": 8.0, "rising": 12.0, "neutral": 6.0}
        self._ball_angle += es.get("velocity", 0.0) * spin_rates.get(mode, 12.0)

        if mode == "flow":
            # ── ROCKET BOOST BALL ─────────────────────────────────────
            # Pulsing cyan rings — large, intense
            pulse = 0.5 + 0.5 * math.sin(self._tick * 0.3)   # 0→1 pulse
            for r in range(60, 8, -6):
                t = (60 - r) / 52
                alpha = int((8 + 50 * t) * (0.7 + 0.3 * pulse) * strength)
                g = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(g, (30, int(200 + 55 * t), int(200 + 55 * t), alpha),
                                   (r, r), r)
                self.screen.blit(g, (ball_x - r, ball_y - r))
            # Bright white core
            pygame.draw.circle(self.screen, (220, 255, 255), (ball_x, ball_y), 14)
            pygame.draw.circle(self.screen, self.C_WHITE,    (ball_x, ball_y), 10)

        elif mode == "falling":
            # ── COMET NUCLEUS ─────────────────────────────────────────
            # Large orange/red halo — comet head
            for r in range(52, 8, -6):
                t = (52 - r) / 44
                alpha = int((10 + 55 * t) * strength)
                gc = (255, int(120 * (1 - t)), 0, alpha)
                g = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(g, gc, (r, r), r)
                self.screen.blit(g, (ball_x - r, ball_y - r))
            # Orange core → white nucleus
            pygame.draw.circle(self.screen, (255, 140,  40), (ball_x, ball_y), 13)
            pygame.draw.circle(self.screen, (255, 220, 180), (ball_x, ball_y), 8)
            pygame.draw.circle(self.screen, self.C_WHITE,    (ball_x, ball_y), 5)

        elif mode == "rising":
            # ── SLOW CLIMB BALL ───────────────────────────────────────
            # Soft white glow, modest size
            for r in range(32, 8, -6):
                t = (32 - r) / 24
                alpha = int((5 + 25 * t) * strength)
                g = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(g, (200, 215, 230, alpha), (r, r), r)
                self.screen.blit(g, (ball_x - r, ball_y - r))
            pygame.draw.circle(self.screen, self.C_WHITE, (ball_x, ball_y), 11)

        else:
            # ── NEUTRAL BALL ──────────────────────────────────────────
            # Dim white, minimal glow, gentle
            for r in range(26, 8, -6):
                t = (26 - r) / 18
                alpha = int((4 + 18 * t) * max(strength, 0.3))
                g = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(g, (170, 180, 200, alpha), (r, r), r)
                self.screen.blit(g, (ball_x - r, ball_y - r))
            pygame.draw.circle(self.screen, (210, 215, 225), (ball_x, ball_y), 11)

        # Spiral etched in ball (all modes)
        spiral_pts = []
        turns, steps, max_r = 1.8, 42, 9
        for i in range(steps + 1):
            t = i / steps
            theta = t * turns * 2 * math.pi + self._ball_angle
            r_sp = t * max_r
            spiral_pts.append((int(ball_x + r_sp * math.cos(theta)),
                                int(ball_y + r_sp * math.sin(theta))))
        if len(spiral_pts) > 1:
            pygame.draw.lines(self.screen, self.C_BG, False, spiral_pts, 1)

    # ── HUD layout ────────────────────────────────────────────────────────────

    def _draw_hud(self, es: dict):
        W, H = self.W, self.H
        score = int(es["score"])
        strength_int = int(es["strength"] * 100)
        self._top_strength = max(self._top_strength, strength_int)
        elapsed = es["elapsed"]
        duration = self.cfg["duration"]

        # ── top-left: label + score ───────────────────────────────────────────
        self.screen.blit(
            self.f_sm.render("CURRENT FRONTAL EEG", True, self.C_LABEL), (22, 18))
        self.screen.blit(
            self.f_sm.render("ACTIVATION FOCUS (FP1/FP2)", True, self.C_LABEL), (22, 34))

        # large score
        s_surf = self.f_xl.render(str(score), True, self.C_WHITE)
        pts_surf = self.f_md.render(" pts", True, self.C_LABEL)
        self.screen.blit(s_surf, (22, 52))
        self.screen.blit(pts_surf, (22 + s_surf.get_width(), 52 + s_surf.get_height() - pts_surf.get_height() - 4))

        # ── top-right: timer + progress bar ──────────────────────────────────
        mins_e = int(elapsed) // 60
        secs_e = int(elapsed) % 60
        mins_t = int(duration) // 60
        secs_t = int(duration) % 60
        tr_x = W - int(W * 0.375)        # right block left edge
        self.screen.blit(
            self.f_sm.render("TARGET STABILITY SEQUENCE:", True, self.C_LABEL),
            (tr_x, 18))
        dur_text = f"[duration {mins_e}:{secs_e:02d} / {mins_t}:{secs_t:02d} min]"
        self.screen.blit(
            self.f_lg.render(dur_text, True, self.C_WHITE),
            (tr_x, 34))

        # progress bar (purple→cyan)
        frac = min(elapsed / duration, 1.0)
        bar = pygame.Rect(tr_x, 68, W - tr_x - int(W * 0.012), 6)
        _gradient_bar(self.screen, bar, frac, radius=3)

        # ── bottom two panels ─────────────────────────────────────────────────
        PH   = int(H * 0.165)              # ~119 at 720p, scales with H
        py   = H - PH - int(H * 0.02)
        mid  = int(W * 0.492)              # panel split point
        left_rect  = pygame.Rect(int(W * 0.011), py, mid - int(W * 0.022), PH)
        right_rect = pygame.Rect(mid, py, W - mid - int(W * 0.011), PH)
        _panel(self.screen, left_rect)
        _panel(self.screen, right_rect)

        # — LEFT panel —
        lx, ly = left_rect.left + 16, left_rect.top + 12
        lw = left_rect.width - 32
        flow_active = es.get("flow_active", False)
        vel = es.get("strength_velocity", 0.0)

        # ── Row: section label + state tag (right-aligned) ───────────────
        sec_lbl = self.f_sm.render("FLOW TRACKER", True, self.C_LABEL)
        self.screen.blit(sec_lbl, (lx, ly))

        # State tag: FOCUSED / RECOVERING / DISTRACTED
        if flow_active:
            tag_text, tag_color = "● IN FLOW", (64, 224, 208)
        elif vel > 0.05:
            tag_text, tag_color = "▲ RECOVERING", (130, 220, 100)
        elif vel < -0.05:
            tag_text, tag_color = "▼ DISTRACTED", (210, 70, 50)
        else:
            tag_text, tag_color = "— NEUTRAL", self.C_DIM
        tag_surf = self.f_sm.render(tag_text, True, tag_color)
        self.screen.blit(tag_surf, (lx + lw - tag_surf.get_width(), ly))
        ly += sec_lbl.get_height() + 6

        # Thin separator
        pygame.draw.line(self.screen, (55, 65, 105),
                         (lx, ly), (lx + lw, ly), 1)
        ly += 8

        # LONGEST FLOW chip
        chip_text = f"{int(es['longest_flow'])}s"
        chip_surf = self.f_lg.render(chip_text, True, self.C_WHITE)
        chip_alpha = pygame.Surface(chip_surf.get_size(), pygame.SRCALPHA)
        chip_alpha.blit(chip_surf, (0, 0))
        chip_alpha.set_alpha(155)
        lf_lbl = self.f_sm.render("LONGEST FLOW", True, self.C_LABEL)
        self.screen.blit(lf_lbl, (lx, ly))
        self.screen.blit(chip_alpha, (lx, ly + lf_lbl.get_height() + 2))

        # Strength + Stability stacked on the right half
        col2_x = lx + lw // 2
        self._hud_row(col2_x, ly, lw // 2,
                      "STRENGTH:", f"{strength_int}/100", strength_int / 100)
        self._hud_row(col2_x, ly + 46, lw // 2,
                      "STABILITY", f"{es['stability'] * 10:.1f}/10", es["stability"])

        # — RIGHT panel: Calibrative Adaptive —
        rx, ry = right_rect.left + 16, right_rect.top + 12
        rw = right_rect.width - 32

        phase = es.get("phase", 1)
        phase_cfg = {
            1: ("● CALIBRATING", (217, 160,  30)),
            2: ("● ADAPTING",    ( 64, 224, 208)),
            3: ("● PEAK PUSH",   (180, 100, 255)),
        }
        phase_label, phase_color = phase_cfg.get(phase, phase_cfg[2])

        # ── Row: section label + phase tag (right-aligned) ──────────────
        sec_lbl = self.f_sm.render("ADAPTIVE ENGINE", True, self.C_LABEL)
        self.screen.blit(sec_lbl, (rx, ry))
        phase_surf = self.f_sm.render(phase_label, True, phase_color)
        self.screen.blit(phase_surf, (rx + rw - phase_surf.get_width(), ry))
        ry += sec_lbl.get_height() + 6

        # Thin separator
        pygame.draw.line(self.screen, (55, 65, 105), (rx, ry), (rx + rw, ry), 1)
        ry += 8

        # ── Big gain number (left half) + two hud_rows (right half) ─────
        gain    = es.get("alpha_gain", 1.0)
        tgt_pct = int(es.get("target_percentile", 65))

        gain_lbl  = self.f_sm.render("SENSITIVITY", True, self.C_LABEL)
        gain_surf = self.f_lg.render(f"×{gain:.2f}", True, self.C_WHITE)
        gain_alpha = pygame.Surface(gain_surf.get_size(), pygame.SRCALPHA)
        gain_alpha.blit(gain_surf, (0, 0))
        gain_alpha.set_alpha(155)
        self.screen.blit(gain_lbl,  (rx, ry))
        self.screen.blit(gain_alpha, (rx, ry + gain_lbl.get_height() + 2))

        col2_x = rx + rw // 2
        top_frac = min(self._top_strength / 200, 1.0)
        self._hud_row(col2_x, ry, rw // 2,
                      "SESSION PEAK:", f"{self._top_strength}/200", top_frac)

        vel = es.get("strength_velocity", 0.0)
        if vel < -0.05:
            vel_label, vel_color = "VELOCITY  ▼", (220, 70, 50)
        elif vel > 0.05:
            vel_label, vel_color = "VELOCITY  ▲", (100, 220, 120)
        else:
            vel_label, vel_color = "VELOCITY  —", self.C_LABEL
        vel_lbl_surf = self.f_sm.render(vel_label, True, vel_color)
        vel_val_surf = self.f_sm.render(f"{vel:+.2f} u/s", True, self.C_WHITE)
        self.screen.blit(vel_lbl_surf, (col2_x, ry + 46))
        self.screen.blit(vel_val_surf,
                         (col2_x + rw // 2 - vel_val_surf.get_width(), ry + 46))
        vel_bar = pygame.Rect(col2_x, ry + 64, rw // 2, 8)
        if vel < -0.05:
            drop_w = int(vel_bar.width * min(abs(vel) * 5, 1.0))
            pygame.draw.rect(self.screen, (22, 25, 42), vel_bar, border_radius=3)
            if drop_w > 0:
                pygame.draw.rect(self.screen, (200, 55, 40),
                                 (vel_bar.left, vel_bar.top, drop_w, vel_bar.height),
                                 border_radius=3)
        else:
            _gradient_bar(self.screen, vel_bar, min(abs(vel) * 5, 1.0))

    def _hud_row(self, x, y, w, label, value, frac):
        """Labelled metric row with a gradient bar."""
        lbl = self.f_sm.render(label, True, self.C_LABEL)
        val = self.f_sm.render(value, True, self.C_WHITE)
        self.screen.blit(lbl, (x, y))
        self.screen.blit(val, (x + w - val.get_width(), y))
        bar = pygame.Rect(x, y + 18, w, 8)
        _gradient_bar(self.screen, bar, frac)

    # ── screens ───────────────────────────────────────────────────────────────

    def _base_frame(self):
        self.screen.fill(self.C_BG)
        self._ensure_grid()
        self.screen.blit(self._grid_surf, (0, 0))

    def _draw_waiting(self):
        self._base_frame()
        W, H = self.W, self.H

        title = self.f_lg.render("FRONTAL NEUROTRAINER", True, self.C_WHITE)
        sub = self.f_md.render(self.cfg["label"], True, self.C_LABEL)
        self.screen.blit(title, title.get_rect(center=(W // 2, H // 2 - 50)))
        self.screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 - 14)))

        latest = self.eeg_thread.get_latest()
        if latest is None:
            msg = self.f_md.render("SEARCHING FOR EEG STREAM…", True, self.C_LABEL)
        else:
            msg = self.f_lg.render("PRESS  SPACE  TO START", True, self.C_CYAN)
        self.screen.blit(msg, msg.get_rect(center=(W // 2, H // 2 + 30)))

        hint = self.f_sm.render("ESC — quit   |   F — fullscreen", True, self.C_DIM)
        self.screen.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 70)))

    def _draw_running(self, es: dict):
        self._base_frame()
        self._tick += 1

        self._wave_history.append(es["position"])
        mode = self._motion_mode(es)

        # Neutral uses low sigma so EEG oscillation is visible as natural wave bumps;
        # flow uses slightly tighter sigma for crisper booster shape.
        sigma_map = {"flow": 6.0, "falling": 10.0, "rising": 10.0, "neutral": 3.5}
        wave_ys, trail_smooth = self._get_wave_ys(sigma=sigma_map[mode])
        ball_x = int(self.W * self.BALL_X_FRAC)

        self._draw_trail(wave_ys, ball_x, mode, es)
        self._draw_ball(wave_ys, es, mode)
        self._draw_hud(es)

    def _draw_results(self, es: dict):
        self._base_frame()
        W, H = self.W, self.H
        cy = H // 2 - 160

        def line(text, font, color):
            nonlocal cy
            s = font.render(text, True, color)
            self.screen.blit(s, s.get_rect(center=(W // 2, cy)))
            cy += s.get_height() + 14

        line("SESSION COMPLETE", self.f_xl, self.C_WHITE)
        cy += 8
        line(f"SCORE   {int(es['score'])}", self.f_lg, self.C_CYAN)
        line(f"TOP STRENGTH   {self._top_strength}/100", self.f_md, self.C_WHITE)
        line(f"LONGEST FLOW   {es['longest_flow']:.1f} s", self.f_md, self.C_LABEL)
        line(f"TOTAL FLOW     {es['flow_time_total']:.1f} s", self.f_md, self.C_LABEL)
        line(f"FLOW ENTRIES   {es['flow_entries']}", self.f_md, self.C_LABEL)
        cy += 20
        line("SPACE — play again     ESC — quit", self.f_sm, self.C_DIM)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            self.clock.tick(int(1 / DT))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if (self.state == "running"
                            and self._pause_rect
                            and self._pause_rect.collidepoint(event.pos)):
                        self._paused = not self._paused
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p and self.state == "running":
                        self._paused = not self._paused
                    elif event.key == pygame.K_f:
                        self._fullscreen = not self._fullscreen
                        if self._fullscreen:
                            # (0,0) lets pygame pick native desktop resolution
                            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            self.screen = pygame.display.set_mode((1280, 720))
                        # Read actual pixel size back so all layout math is correct
                        self.W, self.H = self.screen.get_size()
                        self._grid_surf = None   # rebuild for new size
                    elif event.key == pygame.K_SPACE:
                        if self.state == "waiting":
                            if self.eeg_thread.get_latest() is not None:
                                self.state = "running"
                        elif self.state == "results":
                            self.engine = FocusGameEngine(self.mode)
                            self._wave_history = deque([0.5] * self.WAVE_LEN,
                                                       maxlen=self.WAVE_LEN)
                            self._ball_angle = 0.0
                            self._top_strength = 0
                            self._paused = False
                            self.state = "running"

            if self.state == "waiting":
                self._draw_waiting()
            elif self.state == "running":
                if not self._paused:
                    features = self.eeg_thread.get_latest()
                    if features is not None:
                        self.engine_state = self.engine.tick(features)
                if self.engine_state:
                    self._draw_running(self.engine_state)
                    if self._paused:
                        ov = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
                        ov.fill((0, 0, 0, 120))
                        self.screen.blit(ov, (0, 0))
                        msg = self.f_lg.render("PAUSED  —  P TO RESUME", True, self.C_WHITE)
                        self.screen.blit(msg, msg.get_rect(center=(self.W // 2, self.H // 2)))
                if not self._paused and self.engine_state.get("elapsed", 0) >= self.cfg["duration"]:
                    self.state = "results"
            elif self.state == "results":
                self._draw_results(self.engine_state)

            pygame.display.flip()

        self.eeg_thread.running = False
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG Focus Training Game")
    parser.add_argument("--mode", choices=["2min", "5min"], default="5min",
                        help="Session length / difficulty preset")
    parser.add_argument("--no-lsl", action="store_true",
                        help="Use synthetic EEG (no hardware required)")
    args = parser.parse_args()

    game = FocusGame(mode=args.mode, use_lsl=not args.no_lsl)
    game.run()


if __name__ == "__main__":
    main()
