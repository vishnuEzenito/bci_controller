import numpy as np
import time
import threading
import queue
from collections import deque
from pylsl import resolve_streams, StreamInlet
from scipy.signal import butter, filtfilt, welch
import pygame
import pyautogui
from pynput.keyboard import Key

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

pygame.init()
font       = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 18)
title_font = pygame.font.Font(None, 28)

MONITOR_WIDTH, MONITOR_HEIGHT = 700, 640
monitor_screen = pygame.display.set_mode((MONITOR_WIDTH, MONITOR_HEIGHT))
pygame.display.set_caption("EEG Command Executor")

WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
GREEN      = (50,  255, 50)
RED        = (255, 50,  50)
BLUE       = (50,  100, 255)
GRAY       = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW     = (255, 255, 0)
ORANGE     = (255, 165, 0)
PURPLE     = (128, 0,   128)
DARK_BLUE  = (0,   0,   139)
TEAL       = (0,   180, 180)
PINK       = (255, 100, 180)


class UniversalEEGCommandExecutor:
    def __init__(self):
        self.DEBUG       = True
        self.debug_counter = 0
        self.DEBUG_EVERY = 50

        # EEG state
        self.connection_status = "Disconnected"
        self.current_state     = "Neutral"
        self.state_strength    = 0.0
        self.alpha_power       = 0.0
        self.beta_power        = 0.0
        self.theta_power       = 0.0
        self.gamma_power       = 0.0

        # Per-channel alpha (needed for FAA)
        self.fp1_alpha = 0.0
        self.fp2_alpha = 0.0

        # Derived passive metrics
        self.faa_stress       = 0.5   # 0 = calm, 1 = stressed
        self.engagement_index = 0.0   # beta / (alpha + theta)

        # Alpha blocking detector
        self.alpha_power_history    = deque(maxlen=125)   # ~0.5 s at 250 Hz
        self.alpha_baseline_history = deque(maxlen=1000)  # ~4 s at 250 Hz (adapts after state switch)
        self.alpha_block_detected   = False
        self.alpha_block_flash_until = 0.0
        self.last_alpha_block_time  = 0.0
        self.alpha_block_cooldown   = 3.0  # longer cooldown dampens false triggers on state transitions
        self.alpha_blocks_detected  = 0

        # Key commands
        self.command_config = {
            "Focus": {
                "key": "space",
                "type": "hold",
                "enabled": True,
                "min_strength": 0.3,
                "description": "Hold SPACE while focusing"
            },
            "Relax": {
                "key": "r",
                "type": "press",
                "enabled": True,
                "min_strength": 0.5,
                "description": "Press R when relaxed"
            },
            "Blink": {
                "key": "j",
                "type": "press",
                "enabled": False,
                "cooldown": 0.3,
                "description": "Press J on blink"
            },
            "AlphaBlock": {
                "key": "k",
                "type": "press",
                "enabled": True,
                "description": "Press K on alert/arousal"
            },
        }

        # Blink detection
        self.blink_detected      = False
        self.last_blink_time     = 0
        self.blink_threshold     = 500
        self.blink_cooldown      = 0.3
        self.blink_baseline      = 0
        self.blink_std           = 1.0
        self.min_blink_amplitude = 80
        self.blink_sensitivity   = 1.2
        self.baseline_samples    = deque(maxlen=500)
        self.last_baseline_update = 0
        self.baseline_update_interval = 1.0

        # Key state tracking
        self.key_states       = {}
        self.last_focus_state = False
        self.last_relax_time  = 0

        # Signal quality
        self.signal_quality = 0.0
        self.noise_level    = 0.0

        # Processing parameters
        self.fs            = 250
        self.window_length = 3
        self.min_samples   = self.fs

        # State smoothing
        self.state_history = deque(maxlen=5)
        self.blink_buffer  = deque(maxlen=int(self.fs * 0.5))

        # Statistics
        self.commands_sent  = 0
        self.focus_time     = 0
        self.relax_time     = 0
        self.blinks_detected = 0
        self.session_start  = time.time()

        # Thread control
        self.running    = True
        self.data_queue = queue.Queue()

        # Calibration
        self.calibration_mode = False
        self.calibration_data = {"focus": [], "relax": [], "neutral": []}
        self.user_thresholds  = {"focus_beta": 0.3, "relax_alpha": 0.3}

        # Blink threshold stability
        self.baseline_window          = 500
        self.threshold_history        = deque(maxlen=50)
        self.calibrated_threshold     = None
        self.last_stable_threshold    = None
        self.threshold_stability_factor = 0.95
        self.min_threshold            = 20
        self.max_threshold            = 2000
        self.baseline_drift_limit     = 0.2
        self.sensitivity_step         = 0.05
        self.min_sensitivity          = 0.5
        self.max_sensitivity          = 2.0
        self.target_application       = "Any Application"

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def dbg(self, tag, msg):
        if self.DEBUG:
            print(f"[DEBUG:{tag}] {msg}")

    # ------------------------------------------------------------------
    # Key commands
    # ------------------------------------------------------------------

    def send_key_command(self, command_type, strength=1.0):
        if command_type not in self.command_config:
            return

        config = self.command_config[command_type]
        if not config["enabled"]:
            if command_type == "Focus" and self.last_focus_state:
                pyautogui.keyUp(config["key"])
                self.last_focus_state = False
            return

        if command_type == "Focus":
            if strength >= config["min_strength"] and not self.last_focus_state:
                pyautogui.keyDown(config["key"])
                self.last_focus_state = True
                self.commands_sent += 1
                print(f"Focus ON: Holding {config['key']} (strength: {strength:.2f})")
            elif (strength < config["min_strength"] or command_type != "Focus") and self.last_focus_state:
                pyautogui.keyUp(config["key"])
                self.last_focus_state = False
                print(f"Focus OFF: Released {config['key']}")
            return

        if command_type == "Relax":
            if time.time() - self.last_relax_time < 1.0:
                return
            self.last_relax_time = time.time()

        if command_type == "AlphaBlock":
            pyautogui.press(config["key"])
            self.commands_sent += 1
            print(f"Alpha Block: Pressed {config['key']}")
            return

        try:
            pyautogui.press(config["key"])
            self.commands_sent += 1
            print(f"{command_type}: Pressed {config['key']}")
        except Exception as e:
            print(f"Key command error: {e}")

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return b, a

    def butter_bandstop(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='bandstop')
        return b, a

    def apply_filter(self, data, b, a):
        try:
            return filtfilt(b, a, data, axis=0)
        except:
            return data

    def robust_welch(self, eeg, fs):
        try:
            if len(eeg) < fs:
                return None, None
            nperseg = min(fs * 2, len(eeg) // 4)
            f, psd = welch(eeg, fs, nperseg=nperseg, noverlap=nperseg // 2)
            valid = np.isfinite(psd)
            return f[valid], psd[valid]
        except:
            return None, None

    def bandpower(self, f, psd, band):
        idx = np.logical_and(f >= band[0], f <= band[1])
        if np.sum(idx) == 0:
            return 0
        # Use trapezoid for numpy 2.0+ (trapz was deprecated)
        try:
            return np.trapezoid(psd[idx], f[idx])
        except AttributeError:
            return np.trapz(psd[idx], f[idx])

    def calculate_signal_quality(self, data):
        try:
            qualities, noises = [], []
            for ch in range(2):
                ch_data = data[:, ch]
                signal_range = np.ptp(ch_data)
                if signal_range < 2:
                    return 0.1, 0.9
                signal_std = np.std(ch_data)
                z_scores = np.abs((ch_data - np.mean(ch_data)) / (signal_std + 1e-6))
                artifact_ratio = np.mean(z_scores > 4)
                variance_score = 0.8 if 10 <= signal_std <= 150 else max(0.2, 1.0 - abs(signal_std - 80) / 100)
                noise_level = min(1.0, artifact_ratio + (1.0 if signal_std > 150 else 0.0))
                qualities.append(variance_score * (1 - artifact_ratio))
                noises.append(noise_level)
            quality = np.mean(qualities)
            noise   = np.mean(noises)
            try:
                correlation = abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
                if 0.3 <= correlation <= 0.8:
                    quality = min(1.0, quality + 0.1)
            except:
                pass
            return max(0.1, min(1.0, quality)), min(1.0, max(0.0, noise))
        except:
            return 0.5, 0.5

    # ------------------------------------------------------------------
    # Blink detection
    # ------------------------------------------------------------------

    def detect_blink(self, raw_data):
        current_time = time.time()
        if len(raw_data) < 10:
            return False

        ch1_signal = raw_data[:, 0]
        ch2_signal = raw_data[:, 1]

        if self.blink_baseline is None:
            self.blink_baseline = np.median(ch1_signal + ch2_signal) / 2

        combined_signal = (ch1_signal + ch2_signal) / 2
        if len(combined_signal) > 0:
            self.baseline_samples.extend(combined_signal)

        if len(self.baseline_samples) < 10:
            return False

        try:
            window = max(3, int(5 * self.blink_sensitivity))
            hwin = np.hanning(window) / np.sum(np.hanning(window))
            ch1_smooth = np.convolve(ch1_signal - np.mean(ch1_signal), hwin, mode='valid')
            ch2_smooth = np.convolve(ch2_signal - np.mean(ch2_signal), hwin, mode='valid')
            common_mode = (ch1_smooth + ch2_smooth) / 2

            adjusted_threshold = self.min_blink_amplitude / self.blink_sensitivity
            peak_amplitude = np.max(np.abs(common_mode - self.blink_baseline))
            amplitude_score = min(1.0, peak_amplitude / adjusted_threshold)

            shape_score = 0
            if len(common_mode) > 10:
                derivative  = np.diff(common_mode)
                shape_score = min(1.0, (abs(np.max(derivative)) + abs(np.min(derivative))) / (adjusted_threshold * 0.5))

            correlation = np.corrcoef(ch1_smooth, ch2_smooth)[0, 1]
            min_corr    = 0.5 - (self.blink_sensitivity - 1.0) * 0.2
            correlation_score = max(0, (correlation - min_corr) / (1 - min_corr)) if not np.isnan(correlation) else 0

            confidence     = amplitude_score * 0.4 + shape_score * 0.4 + correlation_score * 0.2
            min_confidence = 0.6 / self.blink_sensitivity

            if confidence > min_confidence and current_time - self.last_blink_time > self.blink_cooldown:
                self.last_blink_time = current_time
                return True
            return False
        except Exception as e:
            print(f"Blink detection error: {e}")
            return False

    def calibrate_blink(self, duration=5):
        self.baseline_samples.clear()
        self.threshold_history.clear()
        baseline_data = []
        start_time = time.time()
        print("\nRecording baseline — don't blink!")
        while time.time() - start_time < duration:
            print(f"  {duration - (time.time() - start_time):.1f}s remaining...", end='\r')
            if self.baseline_samples:
                baseline_data.extend(list(self.baseline_samples))
            time.sleep(0.1)
        if baseline_data:
            self.blink_baseline       = np.median(baseline_data)
            mad                       = np.median(np.abs(np.array(baseline_data) - self.blink_baseline))
            self.blink_std            = mad * 1.4826
            self.calibrated_threshold = max(self.min_threshold, self.blink_std * 2.5)
            self.last_stable_threshold = self.calibrated_threshold
            self.min_blink_amplitude  = self.calibrated_threshold
            for _ in range(self.threshold_history.maxlen):
                self.threshold_history.append(self.calibrated_threshold)
            print(f"\nCalibration done — threshold: {self.calibrated_threshold:.2f}")
        else:
            print("\nCalibration failed — no data collected")

    def adjust_sensitivity(self, increase=True):
        try:
            old_sens = self.blink_sensitivity
            if increase:
                step    = self.sensitivity_step * (2 - old_sens)
                new_sens = min(self.max_sensitivity, old_sens + step)
            else:
                step    = self.sensitivity_step * old_sens
                new_sens = max(self.min_sensitivity, old_sens - step)
            if new_sens != old_sens:
                base_thresh = max(self.min_threshold, self.calibrated_threshold or self.min_blink_amplitude)
                new_thresh  = max(self.min_threshold, min(self.max_threshold, base_thresh / new_sens))
                self.blink_sensitivity = new_sens
                print(f"Blink sensitivity: {old_sens:.2f} → {new_sens:.2f}  (threshold: {new_thresh:.1f})")
        except Exception as e:
            print(f"Sensitivity adjustment error: {e}")

    # ------------------------------------------------------------------
    # Mental state classification
    # ------------------------------------------------------------------

    def classify_mental_state(self, alpha_avg, beta_avg, theta_avg):
        total_power = alpha_avg + beta_avg
        if total_power < 0.1:
            if self.last_focus_state:
                self.send_key_command("Focus", 0.0)
            return "Neutral", 0.1
        if beta_avg > alpha_avg * 1.1 and beta_avg > 0.3:
            return "Focus", min(beta_avg / (alpha_avg + beta_avg), 1.0)
        if self.last_focus_state:
            self.send_key_command("Focus", 0.0)
        if alpha_avg > beta_avg * 1.4 and alpha_avg > 0.3:
            return "Relax", min(alpha_avg / (alpha_avg + beta_avg), 1.0)
        return "Neutral", 0.3

    def adaptive_classify_mental_state(self, alpha_avg, beta_avg, theta_avg):
        focus_threshold = self.user_thresholds.get("focus_beta", 0.5)
        relax_threshold = self.user_thresholds.get("relax_alpha", 0.3)
        total_power = alpha_avg + beta_avg
        if total_power < 0.1:
            return "Neutral", 0.1
        if beta_avg > alpha_avg * 1.1 and beta_avg > focus_threshold:
            return "Focus", min(beta_avg / (alpha_avg + beta_avg), 1.0)
        if alpha_avg > beta_avg * 1.4 and alpha_avg > relax_threshold:
            return "Relax", min(alpha_avg / (alpha_avg + beta_avg), 1.0)
        return "Neutral", 0.3

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def start_calibration(self, state_type, duration=10):
        print(f"\nCalibrating {state_type} for {duration}s — please maintain state...")
        self.calibration_mode = True
        start_time = time.time()
        calibration_samples = []
        while time.time() - start_time < duration:
            calibration_samples.append({
                'alpha': self.alpha_power,
                'beta':  self.beta_power,
                'theta': self.theta_power,
                'timestamp': time.time()
            })
            time.sleep(0.1)
        self.calibration_mode = False
        if calibration_samples:
            self.calibration_data[state_type.lower()] = calibration_samples
            self.update_user_thresholds()
            print(f"Calibration complete: {len(calibration_samples)} samples collected")
        else:
            print("Calibration failed — no data")

    def update_user_thresholds(self):
        try:
            if self.calibration_data["focus"]:
                focus_betas = [s['beta'] for s in self.calibration_data["focus"]]
                self.user_thresholds["focus_beta"] = np.mean(focus_betas) * 0.8
            if self.calibration_data["relax"]:
                relax_alphas = [s['alpha'] for s in self.calibration_data["relax"]]
                self.user_thresholds["relax_alpha"] = np.mean(relax_alphas) * 0.8
            print(f"Updated thresholds — focus β: {self.user_thresholds['focus_beta']:.3f}, "
                  f"relax α: {self.user_thresholds['relax_alpha']:.3f}")
        except Exception as e:
            print(f"Threshold update error: {e}")

    # ------------------------------------------------------------------
    # EEG processing thread
    # ------------------------------------------------------------------

    def eeg_processing_thread(self):
        try:
            print("Looking for EEG stream (ORIC)...")
            streams     = resolve_streams(wait_time=5.0)
            oric_stream = next((s for s in streams if s.name() == 'ORIC'), None)
            if oric_stream is None:
                print("No ORIC stream found!")
                self.connection_status = "No Stream Found"
                return

            inlet = StreamInlet(oric_stream, max_chunklen=1)
            print("Connected to ORIC stream")
            self.connection_status = "Connected"

            buffer          = deque(maxlen=self.window_length * self.fs)
            last_focus_time = time.time()
            last_relax_time = time.time()
            old_state       = "Neutral"

            while self.running:
                try:
                    sample, timestamp = inlet.pull_sample(timeout=1.0)
                    if sample is None:
                        continue

                    if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                        self.dbg("SAMPLE", f"len={len(sample)} first={sample[:4] if len(sample) >= 4 else sample}")

                    buffer.append(sample[:2])  # FP1 (ch0), FP2 (ch1)

                    if len(buffer) >= self.min_samples:
                        data = np.array(buffer)

                        self.signal_quality, self.noise_level = self.calculate_signal_quality(data)

                        # Blink detection on raw data (before filtering)
                        blink_detected = self.detect_blink(data[-int(self.fs * 0.2):])
                        if blink_detected:
                            self.blink_detected = True
                            self.blinks_detected += 1
                            self.send_key_command("Blink")

                        # Bandpass + notch filter
                        try:
                            b, a = self.butter_bandpass(1, 40, self.fs, order=3)
                            data_filtered = self.apply_filter(data, b, a)
                            b, a = self.butter_bandstop(49, 51, self.fs, order=2)
                            data_filtered = self.apply_filter(data_filtered, b, a)
                        except:
                            data_filtered = data

                        # Per-channel PSD
                        ch_results = []
                        for ch in range(2):
                            eeg = data_filtered[:, ch]
                            f, psd = self.robust_welch(eeg, self.fs)
                            if f is not None and psd is not None:
                                try:
                                    theta = self.bandpower(f, psd, [4,  8])
                                    alpha = self.bandpower(f, psd, [8,  12])
                                    beta  = self.bandpower(f, psd, [13, 30])
                                    gamma = self.bandpower(f, psd, [30, 40])
                                    total = theta + alpha + beta + gamma
                                    if total > 0:
                                        ch_results.append({
                                            'theta': theta / total,
                                            'alpha': alpha / total,
                                            'beta':  beta  / total,
                                            'gamma': gamma / total
                                        })
                                except:
                                    continue

                        if len(ch_results) >= 2:
                            # Store per-channel alpha (normalized)
                            self.fp1_alpha = ch_results[0]['alpha']
                            self.fp2_alpha = ch_results[1]['alpha']

                            # Averages used by state classifier and alpha block detector
                            self.alpha_power = (self.fp1_alpha + self.fp2_alpha) / 2
                            self.beta_power  = (ch_results[0]['beta']  + ch_results[1]['beta'])  / 2
                            self.theta_power = (ch_results[0]['theta'] + ch_results[1]['theta']) / 2
                            self.gamma_power = (ch_results[0]['gamma'] + ch_results[1]['gamma']) / 2

                            # --- FAA stress score ---
                            eps = 1e-10
                            faa = np.log(self.fp2_alpha + eps) - np.log(self.fp1_alpha + eps)
                            # faa > 0 → left frontal active (approach/calm), faa < 0 → stress
                            self.faa_stress = float(np.clip((2.0 - faa) / 4.0, 0.0, 1.0))

                            # --- Engagement index ---
                            denom = self.alpha_power + self.theta_power
                            self.engagement_index = float(
                                np.clip(self.beta_power / (denom + eps), 0.0, 1.0)
                            )

                            # --- Alpha blocking detector ---
                            self.alpha_power_history.append(self.alpha_power)
                            self.alpha_baseline_history.append(self.alpha_power)

                            if len(self.alpha_baseline_history) >= 250:
                                alpha_short    = float(np.mean(self.alpha_power_history))
                                alpha_baseline = float(np.mean(self.alpha_baseline_history))
                                ratio = alpha_short / (alpha_baseline + eps)

                                current_time = time.time()
                                if (ratio < 0.4 and
                                        current_time - self.last_alpha_block_time > self.alpha_block_cooldown):
                                    self.alpha_block_detected  = True
                                    self.alpha_blocks_detected += 1
                                    self.alpha_block_flash_until = current_time + 0.5
                                    self.last_alpha_block_time  = current_time
                                    self.send_key_command("AlphaBlock")
                                    if self.DEBUG:
                                        self.dbg("ALPHA_BLOCK", f"ratio={ratio:.3f}")

                            # --- Mental state classification ---
                            if any(self.calibration_data.values()):
                                state, strength = self.adaptive_classify_mental_state(
                                    self.alpha_power, self.beta_power, self.theta_power
                                )
                            else:
                                state, strength = self.classify_mental_state(
                                    self.alpha_power, self.beta_power, self.theta_power
                                )

                            self.state_history.append((state, strength))

                            if len(self.state_history) >= 5:
                                focus_votes = sum(1 for s, _ in self.state_history if s == "Focus")
                                relax_votes = sum(1 for s, _ in self.state_history if s == "Relax")

                                current_time = time.time()

                                if focus_votes >= 3:
                                    if self.current_state != "Focus":
                                        self.current_state  = "Focus"
                                        self.state_strength = np.mean([s for st, s in self.state_history if st == "Focus"])
                                        self.focus_time    += current_time - last_focus_time
                                        last_focus_time     = current_time
                                elif relax_votes >= 3:
                                    if self.current_state == "Focus":
                                        self.send_key_command("Focus", 0.0)
                                    self.current_state  = "Relax"
                                    self.state_strength = np.mean([s for st, s in self.state_history if st == "Relax"])
                                    self.relax_time    += current_time - last_relax_time
                                    last_relax_time     = current_time
                                else:
                                    if self.current_state == "Focus":
                                        self.send_key_command("Focus", 0.0)
                                    self.current_state  = "Neutral"
                                    self.state_strength = 0.3

                                if self.current_state != old_state:
                                    self.send_key_command(self.current_state, self.state_strength)
                                    old_state = self.current_state

                                if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                                    self.dbg("STATE",
                                             f"state={self.current_state} "
                                             f"α={self.alpha_power:.3f} β={self.beta_power:.3f} "
                                             f"stress={self.faa_stress:.2f} ei={self.engagement_index:.2f}")

                    self.debug_counter += 1
                    if self.blink_detected:
                        time.sleep(0.05)
                        self.blink_detected = False

                except Exception as e:
                    print(f"Processing error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"EEG setup error: {e}")
            self.connection_status = f"Error: {str(e)}"

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _draw_bar(self, surface, x, y, w, h, value, color, bg=WHITE):
        pygame.draw.rect(surface, bg,   (x, y, w, h))
        pygame.draw.rect(surface, GRAY, (x, y, w, h), 1)
        fill = int(w * min(max(value, 0.0), 1.0))
        if fill > 0:
            pygame.draw.rect(surface, color, (x, y, fill, h))

    def draw_monitor(self):
        monitor_screen.fill(WHITE)

        # Title
        title = title_font.render("EEG Command Executor — Passive BCI v2", True, DARK_BLUE)
        monitor_screen.blit(title, (10, 10))

        # --- Connection / status row ---
        y = 42
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y, 680, 55), border_radius=5)
        conn_color = GREEN if self.connection_status == "Connected" else RED
        pygame.draw.circle(monitor_screen, conn_color, (25, y + 18), 8)
        monitor_screen.blit(small_font.render(f"EEG: {self.connection_status}", True, BLACK), (38, y + 13))
        monitor_screen.blit(small_font.render(f"Target: {self.target_application}", True, BLACK), (240, y + 13))
        q_color = GREEN if self.signal_quality > 0.7 else ORANGE if self.signal_quality > 0.4 else RED
        monitor_screen.blit(small_font.render(f"Signal: {self.signal_quality:.1%}", True, q_color), (450, y + 13))
        monitor_screen.blit(small_font.render(f"Cmds: {self.commands_sent}", True, BLACK), (38, y + 33))
        sess = time.time() - self.session_start
        monitor_screen.blit(small_font.render(f"Session: {sess:.0f}s", True, BLACK), (240, y + 33))

        # --- Mental state ---
        y += 70
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y, 680, 75), border_radius=5)
        monitor_screen.blit(font.render("Mental State", True, BLACK), (20, y + 8))
        state_colors = {"Focus": GREEN, "Relax": BLUE, "Neutral": GRAY}
        sc = state_colors.get(self.current_state, GRAY)
        pygame.draw.circle(monitor_screen, sc, (550, y + 37), 22)
        monitor_screen.blit(font.render(self.current_state, True, sc), (20, y + 33))
        self._draw_bar(monitor_screen, 200, y + 37, 200, 18, self.state_strength, sc)
        monitor_screen.blit(small_font.render(f"{self.state_strength:.1%}", True, BLACK), (410, y + 39))

        # --- Brain waves + command config ---
        y += 90
        # Left: band powers
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y, 330, 115), border_radius=5)
        monitor_screen.blit(font.render("Brain Waves", True, BLACK), (20, y + 8))
        waves = [
            ("Alpha (8-12 Hz)", self.alpha_power, BLUE),
            ("Beta  (13-30 Hz)", self.beta_power,  RED),
            ("Theta  (4-8 Hz)",  self.theta_power, GREEN),
            ("Gamma (30-40 Hz)", self.gamma_power, PURPLE),
        ]
        for i, (name, power, color) in enumerate(waves):
            wy = y + 32 + i * 20
            monitor_screen.blit(small_font.render(name, True, BLACK), (20, wy))
            self._draw_bar(monitor_screen, 162, wy + 2, 120, 12, power * 3, color)
            monitor_screen.blit(small_font.render(f"{power:.3f}", True, BLACK), (292, wy))

        # Right: key bindings
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (350, y, 340, 115), border_radius=5)
        monitor_screen.blit(font.render("Key Bindings", True, BLACK), (360, y + 8))
        cy = y + 32
        for name, cfg in self.command_config.items():
            sc2 = GREEN if cfg["enabled"] else RED
            pygame.draw.circle(monitor_screen, sc2, (648, cy + 7), 6)
            monitor_screen.blit(
                small_font.render(f"{name}: [{cfg['key'].upper()}]  {cfg['type'] if 'type' in cfg else 'press'}", True, BLACK),
                (360, cy)
            )
            cy += 22

        # --- Passive markers ---
        y += 130
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y, 680, 75), border_radius=5)
        monitor_screen.blit(font.render("Passive Markers", True, BLACK), (20, y + 8))

        # FAA stress bar
        stress_color = RED if self.faa_stress > 0.65 else ORANGE if self.faa_stress > 0.4 else GREEN
        monitor_screen.blit(small_font.render("Stress (FAA)", True, BLACK), (20, y + 35))
        self._draw_bar(monitor_screen, 115, y + 35, 150, 14, self.faa_stress, stress_color)
        monitor_screen.blit(small_font.render(f"{self.faa_stress:.2f}", True, BLACK), (272, y + 35))

        # Engagement index bar
        monitor_screen.blit(small_font.render("Engagement", True, BLACK), (320, y + 35))
        self._draw_bar(monitor_screen, 415, y + 35, 150, 14, self.engagement_index, TEAL)
        monitor_screen.blit(small_font.render(f"{self.engagement_index:.2f}", True, BLACK), (572, y + 35))

        # Alpha block flash indicator
        is_flashing = time.time() < self.alpha_block_flash_until
        alert_color = YELLOW if is_flashing else LIGHT_GRAY
        pygame.draw.circle(monitor_screen, alert_color, (650, y + 22), 14)
        monitor_screen.blit(small_font.render("ALERT", True, BLACK if is_flashing else GRAY), (626, y + 40))
        monitor_screen.blit(small_font.render(f"x{self.alpha_blocks_detected}", True, BLACK), (633, y + 55))

        # Blink flash
        if self.blink_detected:
            pygame.draw.circle(monitor_screen, YELLOW, (615, y + 22), 14)
            monitor_screen.blit(small_font.render("BLINK", True, BLACK), (592, y + 40))

        # --- Statistics ---
        y += 90
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y, 680, 55), border_radius=5)
        monitor_screen.blit(font.render("Session Stats", True, BLACK), (20, y + 8))
        stats = [
            f"Focus: {self.focus_time:.0f}s",
            f"Relax: {self.relax_time:.0f}s",
            f"Blinks: {self.blinks_detected}",
            f"Alerts: {self.alpha_blocks_detected}",
            f"Noise: {self.noise_level:.1%}",
        ]
        for i, stat in enumerate(stats):
            monitor_screen.blit(small_font.render(stat, True, BLACK), (20 + i * 135, y + 33))

        # --- Instructions ---
        y += 65
        instructions = [
            "Focus → SPACE (hold)   Relax → R   Blink → J   Alpha Block → K",
            "Keys: 1/2/3/4=toggle commands  B=blink-calib  C=mental-calib  UP/DN=sensitivity  Q=quit",
        ]
        for i, line in enumerate(instructions):
            c = DARK_BLUE if i == 0 else GRAY
            monitor_screen.blit(small_font.render(line, True, c), (10, y + i * 18))

        # Calibration overlay
        if self.calibration_mode:
            monitor_screen.blit(font.render("CALIBRATING — follow console instructions", True, RED), (10, 38))

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_key_input(self, event):
        if event.key == pygame.K_1:
            self.command_config["Focus"]["enabled"] = not self.command_config["Focus"]["enabled"]
        elif event.key == pygame.K_2:
            self.command_config["Relax"]["enabled"] = not self.command_config["Relax"]["enabled"]
        elif event.key == pygame.K_3:
            self.command_config["Blink"]["enabled"] = not self.command_config["Blink"]["enabled"]
        elif event.key == pygame.K_4:
            self.command_config["AlphaBlock"]["enabled"] = not self.command_config["AlphaBlock"]["enabled"]
        elif event.key == pygame.K_b:
            print("\nBlink calibration: sit still, look straight, don't blink. Starting in 3s...")
            time.sleep(3)
            self.calibrate_blink()
        elif event.key == pygame.K_c:
            def run_calibration():
                print("\nMental state calibration:")
                print("Step 1: FOCUS — concentrate hard for 10s...")
                time.sleep(3)
                self.start_calibration("focus", 10)
                print("Step 2: RELAX — close eyes, breathe slowly for 10s...")
                time.sleep(3)
                self.start_calibration("relax", 10)
                print("Calibration complete.")
            threading.Thread(target=run_calibration, daemon=True).start()
        elif event.key == pygame.K_UP:
            self.adjust_sensitivity(increase=True)
        elif event.key == pygame.K_DOWN:
            self.adjust_sensitivity(increase=False)

    # ------------------------------------------------------------------
    # Cleanup & run
    # ------------------------------------------------------------------

    def cleanup_keys(self):
        try:
            if self.last_focus_state:
                pyautogui.keyUp(self.command_config["Focus"]["key"])
                self.last_focus_state = False
        except:
            pass

    def run(self):
        print("EEG Command Executor — Passive BCI v2")
        print("  Focus → SPACE (hold)  |  Relax → R  |  Blink → J  |  Alpha Block → K")
        print("  FAA Stress and Engagement shown on UI (display only)")

        eeg_thread = threading.Thread(target=self.eeg_processing_thread, daemon=True)
        eeg_thread.start()

        clock = pygame.time.Clock()
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.running = False
                        else:
                            self.handle_key_input(event)
                self.draw_monitor()
                clock.tick(30)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup_keys()
            self.running = False
            pygame.quit()
            print("Controller stopped.")


if __name__ == "__main__":
    try:
        executor = UniversalEEGCommandExecutor()
        executor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        try:
            pyautogui.keyUp('space')
        except:
            pass
