import numpy as np
import time
import threading
import queue
from collections import deque
from pylsl import resolve_streams, StreamInlet
from scipy.signal import butter, filtfilt, welch
import pygame
import pyautogui
import keyboard
from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Key, Listener
import json

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

# Initialize pygame for display only
pygame.init()
font = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 18)
title_font = pygame.font.Font(None, 28)

# Display settings
MONITOR_WIDTH, MONITOR_HEIGHT = 700, 500
monitor_screen = pygame.display.set_mode((MONITOR_WIDTH, MONITOR_HEIGHT))
pygame.display.set_caption("Universal EEG Command Executor")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 255, 50)
RED = (255, 50, 50)
BLUE = (50, 100, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
DARK_BLUE = (0, 0, 139)

class UniversalEEGCommandExecutor:
    def __init__(self):
        # Debug instrumentation
        self.DEBUG = True
        self.debug_counter = 0
        self.DEBUG_EVERY = 50   # print once every 50 samples
        
        # EEG state variables
        self.connection_status = "Disconnected"
        self.current_state = "Neutral"
        self.state_strength = 0.0
        self.alpha_power = 0.0
        self.beta_power = 0.0
        self.theta_power = 0.0
        self.gamma_power = 0.0
        
        # Command mapping configuration - optimized for gaming
        self.command_config = {
            "Focus": {
                "key": "space",
                "type": "hold",  # "press", "hold", "toggle"
                "enabled": True,
                "min_strength": 0.6,  # Increased threshold for stability
                "description": "Hold SPACE while focusing"
            },
            "Relax": {
                "key": "r",
                "type": "press",
                "enabled": True,
                "min_strength": 0.5,  # Higher threshold for relax
                "description": "Press R when relaxed"  
            },
            "Blink": {
                "key": "j",
                "type": "press",
                "enabled": False,
                "cooldown": 0.,
                "description": "Press space on blink"
            }
        }
        
        # Blink detection - optimized for FP1/FP2 dry electrodes
        self.blink_detected = False
        self.last_blink_time = 0
        self.blink_threshold = 500  # Lower threshold for dry electrodes (was 150)
        self.blink_cooldown = 0.3  # Faster cooldown for responsive gaming (was 0.5)
        
        # Update blink detection parameters with more sensitive defaults
        self.blink_baseline = None
        self.blink_std = None
        self.min_blink_amplitude = 20  # Reduced from 30 for more sensitivity
        self.blink_sensitivity = 1.2   # Increased default sensitivity
        self.baseline_samples = deque(maxlen=500)  # 2 seconds at 250Hz
        self.last_baseline_update = 0
        self.baseline_update_interval = 1.0  # Update baseline every second
        
        # Key state tracking
        self.key_states = {}
        self.last_focus_state = False
        self.last_relax_time = 0
        
        # Signal quality metrics
        self.signal_quality = 0.0
        self.noise_level = 0.0
        
        # Processing parameters - MATCH BALL GAME EXACTLY
        self.fs = 250  # Sampling frequency
        self.window_length = 3  # seconds - MATCH BALL GAME (was 2)
        self.min_samples = self.fs  # Minimum samples for analysis
        
        # State smoothing - MATCH BALL GAME EXACTLY
        self.state_history = deque(maxlen=5)  # MATCH BALL GAME
        self.blink_buffer = deque(maxlen=int(self.fs * 0.5))  # 500ms buffer for robust blink detection
        
        # Statistics
        self.commands_sent = 0
        self.focus_time = 0
        self.relax_time = 0
        self.blinks_detected = 0
        self.session_start = time.time()
        
        # Thread control
        self.running = True
        self.data_queue = queue.Queue()
        
        # Add calibration and user adaptation
        self.calibration_mode = False
        self.calibration_data = {"focus": [], "relax": [], "neutral": []}
        self.user_thresholds = {"focus_beta": 0.5, "relax_alpha": 0.3}  # Will be updated during calibration
        
        # Real-time performance tracking
        self.detection_accuracy = {"focus": 0.8, "relax": 0.8, "blink": 0.7}
        self.false_positive_count = {"focus": 0, "relax": 0, "blink": 0}
        
        # Current application info
        self.target_application = "Any Application"
        
        # Update sensitivity defaults
        self.blink_sensitivity = 1.0   # Start with normal sensitivity
        self.min_sensitivity = 0.5     # Don't allow too low sensitivity
        self.max_sensitivity = 2.0     # Don't allow too high sensitivity
        
        # Add new threshold control parameters with proper initialization
        self.baseline_window = 500  # 2 seconds at 250Hz
        self.baseline_samples = deque(maxlen=self.baseline_window)
        self.threshold_history = deque(maxlen=50)  # Longer history for stability
        self.calibrated_threshold = None  # Store initial calibration threshold
        self.last_stable_threshold = None
        self.threshold_stability_factor = 0.95  # Very high stability
        self.min_threshold = 20  # Minimum allowed threshold
        self.max_threshold = 2000  # Maximum allowed threshold
        self.baseline_drift_limit = 0.2  # Max 20% drift from calibration
        self.sensitivity_step = 0.05  # Add missing sensitivity step
        self.blink_baseline = 0  # Initialize with default value
        self.blink_std = 1.0  # Initialize with default value
    
    def dbg(self, tag, msg):
        """Debug logger for pipeline diagnostics"""
        if self.DEBUG:
            print(f"[DEBUG:{tag}] {msg}")
        
    def send_key_command(self, command_type, strength=1.0):
        """Send keyboard command based on configuration"""
        if command_type not in self.command_config:
            return
        
        config = self.command_config[command_type]
        if not config["enabled"]:
            # Always release held keys if command gets disabled
            if command_type == "Focus" and self.last_focus_state:
                pyautogui.keyUp(config["key"])
                self.last_focus_state = False
                print(f"🔓 Released {config['key']} (command disabled)")
            return

        # Handle focus state changes immediately
        if command_type == "Focus":
            if strength >= config["min_strength"] and not self.last_focus_state:
                pyautogui.keyDown(config["key"])
                self.last_focus_state = True
                self.commands_sent += 1
                print(f"🔒 Focus ON: Holding {config['key']} (strength: {strength:.2f})")
            elif (strength < config["min_strength"] or command_type != "Focus") and self.last_focus_state:
                pyautogui.keyUp(config["key"])
                self.last_focus_state = False
                print(f"🔓 Focus OFF: Released {config['key']}")
            return
            
        # Handle other commands
        key = config["key"]
        cmd_type = config["type"]
        
        try:
            if cmd_type == "press":
                # Check cooldown for certain commands
                if command_type == "Relax":
                    if time.time() - self.last_relax_time < 1.0:
                        return
                    self.last_relax_time = time.time()
                
                pyautogui.press(key)
                self.commands_sent += 1
                print(f"🎯 {command_type}: Pressed {key}")
                
        except Exception as e:
            print(f"Error sending key command: {e}")

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandstop(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandstop')
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
            f, psd = welch(eeg, fs, nperseg=nperseg, noverlap=nperseg//2)
            
            valid_idx = np.isfinite(psd)
            return f[valid_idx], psd[valid_idx]
        except:
            return None, None
    
    def bandpower(self, f, psd, band):
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        if np.sum(idx_band) == 0:
            return 0
        return np.trapz(psd[idx_band], f[idx_band])
    
    def calibrate_blink(self, duration=5):
        """Enhanced blink calibration with stability"""
        self.baseline_samples.clear()
        self.threshold_history.clear()
        baseline_data = []
        start_time = time.time()
        
        print("\n🔄 Recording baseline - Don't blink!")
        while time.time() - start_time < duration:
            remaining = duration - (time.time() - start_time)
            print(f"⏳ {remaining:.1f}s remaining...", end='\r')
            if len(self.baseline_samples) > 0:
                baseline_data.extend(list(self.baseline_samples))
            time.sleep(0.1)
        
        if baseline_data:
            # Calculate robust baseline parametersb
            self.blink_baseline = np.median(baseline_data)
            mad = np.median(np.abs(np.array(baseline_data) - self.blink_baseline))
            self.blink_std = mad * 1.4826
            
            # Set initial calibrated threshold
            self.calibrated_threshold = max(self.min_threshold, self.blink_std * 2.5)
            self.last_stable_threshold = self.calibrated_threshold
            self.min_blink_amplitude = self.calibrated_threshold
            
            # Initialize threshold history with calibrated value
            for _ in range(self.threshold_history.maxlen):
                self.threshold_history.append(self.calibrated_threshold)
            
            print(f"\n\n✅ Calibration Results:")
            print(f"• Baseline level: {self.blink_baseline:.2f}")
            print(f"• Noise level: {self.blink_std:.2f}")
            print(f"• Detection threshold: {self.calibrated_threshold:.2f}")
            print(f"• Current sensitivity: {self.blink_sensitivity:.1f}")
            print("\n👉 Now try blinking normally to test")
            print("👉 Use UP/DOWN arrows to adjust sensitivity if needed")
        else:
            print("\n❌ Calibration failed - No data collected")
            print("Make sure EEG device is connected properly")

    def detect_blink(self, raw_data):
        """Improved blink detection with threshold stability"""
        current_time = time.time()
        
        if len(raw_data) < 10:
            return False
            
        # Get both channels
        ch1_signal = raw_data[:, 0]  # FP1
        ch2_signal = raw_data[:, 1]  # FP2
        
        # Ensure baseline exists
        if self.blink_baseline is None:
            self.blink_baseline = np.median(ch1_signal + ch2_signal) / 2
            
        # Update baseline with exponential moving average
        combined_signal = (ch1_signal + ch2_signal) / 2
        if len(combined_signal) > 0:  # Check if data exists
            self.baseline_samples.extend(combined_signal)
        
        # Ensure we have valid baseline data
        if len(self.baseline_samples) < 10:
            return False
            
        try:
            # Enhanced noise reduction with sensitivity-adjusted window
            window = max(3, int(5 * self.blink_sensitivity))  # Adjust window size with sensitivity
            ch1_smooth = np.convolve(ch1_signal - np.mean(ch1_signal), 
                                    np.hanning(window)/np.sum(np.hanning(window)), 
                                    mode='valid')
            ch2_smooth = np.convolve(ch2_signal - np.mean(ch2_signal), 
                                    np.hanning(window)/np.sum(np.hanning(window)), 
                                    mode='valid')
            
            # Calculate differential and common mode
            differential = ch1_smooth - ch2_smooth
            common_mode = (ch1_smooth + ch2_smooth) / 2
            
            # Dynamic threshold calculation with sensitivity
            adjusted_threshold = (
                self.min_blink_amplitude / self.blink_sensitivity  # Lower threshold when sensitivity is higher
            )
            
            # Multi-factor blink detection
            peak_amplitude = np.max(np.abs(common_mode - self.blink_baseline))
            amplitude_score = min(1.0, peak_amplitude / adjusted_threshold)
            
            # Shape analysis with sensitivity-adjusted thresholds
            if len(differential) > 10:
                derivative = np.diff(common_mode)
                pos_slope = np.max(derivative)
                neg_slope = np.min(derivative)
                shape_score = min(1.0, (abs(pos_slope) + abs(neg_slope)) / (adjusted_threshold * 0.5))
            else:
                shape_score = 0
            
            # Correlation analysis with adjusted threshold
            correlation = np.corrcoef(ch1_smooth, ch2_smooth)[0,1]
            min_correlation = 0.5 - (self.blink_sensitivity - 1.0) * 0.2  # Reduce required correlation at higher sensitivity
            correlation_score = max(0, (correlation - min_correlation) / (1 - min_correlation)) if not np.isnan(correlation) else 0
            
            # Combined confidence score
            confidence = (
                amplitude_score * 0.4 +
                shape_score * 0.4 +
                correlation_score * 0.2
            )
            
            # Adjust confidence threshold based on sensitivity
            min_confidence = 0.6 / self.blink_sensitivity
            
            if confidence > min_confidence and current_time - self.last_blink_time > self.blink_cooldown:
                self.last_blink_time = current_time
                return True
            
            return False
            
        except Exception as e:
            print(f"Blink detection error: {e}")
            return False

    def update_baseline_stats(self):
        """Update baseline with drift protection"""
        if len(self.baseline_samples) > self.baseline_window * 0.8:  # Require 80% full buffer
            current_baseline = np.median(self.baseline_samples)
            current_mad = np.median(np.abs(np.array(self.baseline_samples) - current_baseline))
            current_std = current_mad * 1.4826
            
            # Check for excessive drift from calibration
            if self.calibrated_threshold is not None:
                drift_ratio = abs(current_std - self.blink_std) / self.blink_std
                
                if drift_ratio > self.baseline_drift_limit:
                    # Reset to last stable values if drift is too high
                    return
            
            # Update only if values are reasonable
            if self.min_threshold <= current_std * 2.5 <= self.max_threshold:
                self.blink_baseline = current_baseline
                self.blink_std = current_std
                new_threshold = max(self.min_threshold, current_std * 2.5)
                
                # Smooth threshold updates
                if self.last_stable_threshold is not None:
                    self.min_blink_amplitude = (self.last_stable_threshold * self.threshold_stability_factor +
                                              new_threshold * (1 - self.threshold_stability_factor))
                    self.last_stable_threshold = self.min_blink_amplitude
                else:
                    self.min_blink_amplitude = new_threshold
                    self.last_stable_threshold = new_threshold

    def adjust_sensitivity(self, increase=True):
        """Improved sensitivity adjustment with error checking"""
        try:
            old_sens = self.blink_sensitivity
            
            # Use smaller steps (0.05) for finer control
            if increase:
                step = self.sensitivity_step * (2 - old_sens)  # Smaller steps at higher sensitivity
                new_sens = min(self.max_sensitivity, old_sens + step)
            else:
                step = self.sensitivity_step * old_sens  # Smaller steps at lower sensitivity
                new_sens = max(self.min_sensitivity, old_sens - step)
            
            if new_sens != old_sens:
                # Ensure we have valid baseline values
                if self.calibrated_threshold is None:
                    self.calibrated_threshold = self.min_blink_amplitude
                
                # Calculate new threshold safely
                base_threshold = max(self.min_threshold, 
                                   self.calibrated_threshold or self.min_blink_amplitude)
                new_threshold = base_threshold / new_sens
                
                # Apply threshold bounds
                new_threshold = max(self.min_threshold, 
                                  min(self.max_threshold, new_threshold))
                
                # Update sensitivity and log changes
                self.blink_sensitivity = new_sens
                direction = "📈 Increased" if increase else "📉 Decreased"
                print(f"\n{direction} blink sensitivity: {old_sens:.2f} -> {new_sens:.2f}")
                print(f"New detection threshold: {new_threshold:.1f}")
                
                # Update threshold history safely
                if len(self.threshold_history) > 0:
                    self.threshold_history.append(new_threshold)
                
                return new_threshold
            
            return self.min_blink_amplitude / old_sens
            
        except Exception as e:
            print(f"Sensitivity adjustment error: {e}")
            return self.min_blink_amplitude  # Safe fallback

    def classify_mental_state(self, alpha_avg, beta_avg, theta_avg):
        """Robust classification matching ball game logic"""
        # Calculate ratios for more robust classification
        alpha_beta_ratio = alpha_avg / (beta_avg + 1e-6)
        total_power = alpha_avg + beta_avg
        
        # Dynamic thresholds based on total power
        if total_power < 0.1:
            if self.last_focus_state:  # Always release space if signal is too weak
                self.send_key_command("Focus", 0.0)
            return "Neutral", 0.1
        
        # Focus detection with immediate key handling
        if beta_avg > alpha_avg * 1.3 and beta_avg > 0.5:
            strength = min(beta_avg / (alpha_avg + beta_avg), 1.0)
            return "Focus", strength
        
        # If not focusing, always release space
        if self.last_focus_state:
            self.send_key_command("Focus", 0.0)
        
        # Check for relax state
        if alpha_avg > beta_avg * 1.4 and alpha_avg > 0.3:
            strength = min(alpha_avg / (alpha_avg + beta_avg), 1.0)
            return "Relax", strength
        
        return "Neutral", 0.3

    def eeg_processing_thread(self):
        """Main EEG processing loop"""
        try:
            print("🔎 Looking for EEG stream...")
            streams = resolve_streams(wait_time=5.0)
            oric_stream = next((s for s in streams if s.name() == 'ORIC'), None)
            
            if oric_stream is None:
                print("❌ No ORIC stream found!")
                self.connection_status = "No Stream Found"
                return
            
            inlet = StreamInlet(oric_stream, max_chunklen=1)
            print("✅ Connected to EEG stream")
            self.connection_status = "Connected"
            
            buffer = deque(maxlen=self.window_length * self.fs)
            last_focus_time = time.time()
            last_relax_time = time.time()
            old_state = "Neutral"  # Initialize old_state
            
            while self.running:
                try:
                    sample, timestamp = inlet.pull_sample(timeout=1.0)
                    if sample is None: continue
                    
                    # DEBUG: Check sample shape and values
                    if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                        self.dbg("SAMPLE",
                            f"len={len(sample)} "
                            f"first={sample[:4] if len(sample) >= 4 else sample}")
                    
                    buffer.append(sample[:2])  # First 2 channels
                    
                    # DEBUG: Check buffer state
                    if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                        self.dbg("BUFFER", f"buffer_len={len(buffer)}")
                    
                    if len(buffer) >= self.min_samples:
                        data = np.array(buffer)
                        
                        # DEBUG: Check raw signal statistics
                        if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                            self.dbg("RAW_STATS",
                                f"std0={np.std(data[:,0]):.3f} "
                                f"std1={np.std(data[:,1]):.3f} "
                                f"min0={np.min(data[:,0]):.2f} "
                                f"max0={np.max(data[:,0]):.2f}")
                        
                        # Calculate signal quality
                        self.signal_quality, self.noise_level = self.calculate_signal_quality(data)
                        
                        # Blink detection (on RAW data before filtering - critical!)
                        blink_detected = self.detect_blink(data[-int(self.fs * 0.2):])
                        
                        if blink_detected:
                            self.blink_detected = True
                            self.blinks_detected += 1
                            self.send_key_command("Blink")
                        
                        # Apply filtering AFTER blink detection
                        try:
                            # Bandpass filter (1-40 Hz)
                            b, a = self.butter_bandpass(1, 40, self.fs, order=3)
                            data_filtered = self.apply_filter(data, b, a)
                            
                            # Notch filter (49-51 Hz for 50Hz line noise)
                            b, a = self.butter_bandstop(49, 51, self.fs, order=2)
                            data_filtered = self.apply_filter(data_filtered, b, a)
                        except:
                            data_filtered = data
                        
                        # DEBUG: Check filtered signal
                        if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                            self.dbg("FILTER",
                                f"std0_filt={np.std(data_filtered[:,0]):.3f} "
                                f"std1_filt={np.std(data_filtered[:,1]):.3f}")
                        
                        # Process mental state from both channels
                        ch_results = []
                        for ch in range(2):
                            eeg = data_filtered[:, ch]
                            f, psd = self.robust_welch(eeg, self.fs)
                            
                            # DEBUG: Check PSD success
                            if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                                self.dbg("PSD",
                                    f"ch{ch} f={'OK' if f is not None else 'None'} "
                                    f"psd={'OK' if psd is not None else 'None'} "
                                    f"len={len(psd) if psd is not None else 0}")
                            
                            if f is not None and psd is not None:
                                try:
                                    theta = self.bandpower(f, psd, [4, 8])
                                    alpha = self.bandpower(f, psd, [8, 12])
                                    beta = self.bandpower(f, psd, [13, 30])
                                    gamma = self.bandpower(f, psd, [30, 40])
                                    
                                    # DEBUG: Check bandpower values
                                    if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                                        self.dbg("BANDPOWER",
                                            f"ch{ch} θ={theta:.4f} α={alpha:.4f} β={beta:.4f}")
                                    
                                    total = theta + alpha + beta + gamma
                                    if total > 0:
                                        ch_results.append({
                                            'theta': theta / total,
                                            'alpha': alpha / total,
                                            'beta': beta / total,
                                            'gamma': gamma / total
                                        })
                                except:
                                    continue
                        
                        # DEBUG: Check if we got valid channel results
                        if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                            self.dbg("CH_RESULTS", f"count={len(ch_results)}")
                        
                        # Average across channels and classify
                        if len(ch_results) >= 2:
                            self.alpha_power = (ch_results[0]['alpha'] + ch_results[1]['alpha']) / 2
                            self.beta_power = (ch_results[0]['beta'] + ch_results[1]['beta']) / 2
                            self.theta_power = (ch_results[0]['theta'] + ch_results[1]['theta']) / 2
                            self.gamma_power = (ch_results[0]['gamma'] + ch_results[1]['gamma']) / 2
                            
                            # Use adaptive classification if calibration data is available
                            if any(self.calibration_data.values()):
                                state, strength = self.adaptive_classify_mental_state(
                                    self.alpha_power, self.beta_power, self.theta_power
                                )
                            else:
                                # Use standard classification (exact ball game logic)
                                state, strength = self.classify_mental_state(
                                    self.alpha_power, self.beta_power, self.theta_power
                                )
                            
                            # Smooth the state classification - EXACT BALL GAME LOGIC
                            self.state_history.append((state, strength))
                            
                            # MATCH BALL GAME: Majority vote with strength weighting (requires 3+ votes)
                            if len(self.state_history) >= 5:
                                focus_votes = sum(1 for s, _ in self.state_history if s == "Focus")
                                relax_votes = sum(1 for s, _ in self.state_history if s == "Relax")
                                
                                current_time = time.time()
                                previous_state = self.current_state  # Store previous state
                                
                                # State transitions with key management
                                if focus_votes >= 3:
                                    if self.current_state != "Focus":
                                        self.current_state = "Focus"
                                        self.state_strength = np.mean([s for st, s in self.state_history if st == "Focus"])
                                        self.focus_time += current_time - last_focus_time
                                        last_focus_time = current_time
                                    
                                elif relax_votes >= 3:
                                    if self.current_state == "Focus":
                                        self.send_key_command("Focus", 0.0)  # Release space when switching to relax
                                    self.current_state = "Relax"
                                    self.state_strength = np.mean([s for st, s in self.state_history if st == "Relax"])
                                    self.relax_time += current_time - last_relax_time
                                    last_relax_time = current_time
                                    
                                else:
                                    if self.current_state == "Focus":
                                        self.send_key_command("Focus", 0.0)  # Release space when switching to neutral
                                    self.current_state = "Neutral"
                                    self.state_strength = 0.3
                                
                                # Send commands only on state change
                                if self.current_state != old_state:
                                    self.send_key_command(self.current_state, self.state_strength)
                                    old_state = self.current_state
                                
                                # DEBUG: Report final state
                                if self.DEBUG and self.debug_counter % self.DEBUG_EVERY == 0:
                                    self.dbg("STATE",
                                        f"state={self.current_state} "
                                        f"α={self.alpha_power:.3f} "
                                        f"β={self.beta_power:.3f}")
                    
                    # Increment debug counter
                    self.debug_counter += 1
                    
                    # Reset blink flag
                    if self.blink_detected:
                        time.sleep(0.05)
                        self.blink_detected = False
                
                except Exception as e:
                    print(f"Processing error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"EEG setup error: {e}")
            self.connection_status = f"Error: {str(e)}"
    
    def draw_monitor(self):
        """Draw the comprehensive monitoring interface"""
        monitor_screen.fill(WHITE)
        
        # Title
        title = title_font.render("Universal EEG Command Executor", True, DARK_BLUE)
        monitor_screen.blit(title, (10, 10))
        
        # Connection status section
        y_pos = 45
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y_pos, 680, 60), border_radius=5)
        
        # EEG Connection
        conn_color = GREEN if self.connection_status == "Connected" else RED
        pygame.draw.circle(monitor_screen, conn_color, (25, y_pos + 20), 8)
        conn_text = small_font.render(f"EEG: {self.connection_status}", True, BLACK)
        monitor_screen.blit(conn_text, (40, y_pos + 15))
        
        # Target application
        app_text = small_font.render(f"Target: {self.target_application}", True, BLACK)
        monitor_screen.blit(app_text, (250, y_pos + 15))
        
        # Signal quality
        quality_color = GREEN if self.signal_quality > 0.7 else ORANGE if self.signal_quality > 0.4 else RED
        quality_text = small_font.render(f"Signal Quality: {self.signal_quality:.1%}", True, quality_color)
        monitor_screen.blit(quality_text, (450, y_pos + 15))
        
        # Commands sent
        cmd_text = small_font.render(f"Commands Sent: {self.commands_sent}", True, BLACK)
        monitor_screen.blit(cmd_text, (40, y_pos + 35))
        
        # Session time
        session_time = time.time() - self.session_start
        time_text = small_font.render(f"Session: {session_time:.0f}s", True, BLACK)
        monitor_screen.blit(time_text, (250, y_pos + 35))
        
        y_pos += 80
        
        # Current Mental State Section
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y_pos, 680, 80), border_radius=5)
        
        state_title = font.render("Current Mental State", True, BLACK)
        monitor_screen.blit(state_title, (20, y_pos + 10))
        
        # State indicator
        state_colors = {"Focus": GREEN, "Relax": BLUE, "Neutral": GRAY}
        state_color = state_colors.get(self.current_state, GRAY)
        
        # Large state display
        pygame.draw.circle(monitor_screen, state_color, (550, y_pos + 40), 25)
        state_text = font.render(self.current_state, True, state_color)
        monitor_screen.blit(state_text, (20, y_pos + 35))
        
        # Strength bar
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(monitor_screen, WHITE, (200, y_pos + 40, bar_width, bar_height))
        pygame.draw.rect(monitor_screen, GRAY, (200, y_pos + 40, bar_width, bar_height), 2)
        fill_width = int(bar_width * self.state_strength)
        pygame.draw.rect(monitor_screen, state_color, (200, y_pos + 40, fill_width, bar_height))
        
        strength_text = small_font.render(f"{self.state_strength:.1%}", True, BLACK)
        monitor_screen.blit(strength_text, (410, y_pos + 42))
        
        y_pos += 100
        
        # Brain Wave Analysis Section
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y_pos, 330, 120), border_radius=5)
        
        waves_title = font.render("Brain Wave Analysis", True, BLACK)
        monitor_screen.blit(waves_title, (20, y_pos + 10))
        
        waves = [
            ("Alpha (8-12 Hz)", self.alpha_power, BLUE),
            ("Beta (13-30 Hz)", self.beta_power, RED),
            ("Theta (4-8 Hz)", self.theta_power, GREEN),
            ("Gamma (30-40 Hz)", self.gamma_power, PURPLE)
        ]
        
        for i, (wave_name, power, color) in enumerate(waves):
            wave_y = y_pos + 35 + i * 20
            
            # Wave name
            wave_text = small_font.render(wave_name, True, BLACK)
            monitor_screen.blit(wave_text, (20, wave_y))
            
            # Power bar
            bar_width = 120
            bar_height = 12
            pygame.draw.rect(monitor_screen, WHITE, (160, wave_y + 2, bar_width, bar_height))
            pygame.draw.rect(monitor_screen, GRAY, (160, wave_y + 2, bar_width, bar_height), 1)
            
            fill_width = int(bar_width * min(power * 3, 1.0))  # Scale for visibility
            pygame.draw.rect(monitor_screen, color, (160, wave_y + 2, fill_width, bar_height))
            
            # Power value
            power_text = small_font.render(f"{power:.3f}", True, BLACK)
            monitor_screen.blit(power_text, (290, wave_y))
        
        # Command Configuration Section
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (350, y_pos, 340, 120), border_radius=5)
        
        config_title = font.render("Command Configuration", True, BLACK)
        monitor_screen.blit(config_title, (360, y_pos + 10))
        
        config_y = y_pos + 35
        for command, config in self.command_config.items():
            # Command name and key
            cmd_text = small_font.render(f"{command}: {config['key'].upper()}", True, BLACK)
            monitor_screen.blit(cmd_text, (360, config_y))
            
            # Status indicator
            status_color = GREEN if config['enabled'] else RED
            pygame.draw.circle(monitor_screen, status_color, (650, config_y + 8), 6)
            
            # Type and threshold
            details = f"({config['type']}"
            if 'min_strength' in config:
                details += f", >{config['min_strength']:.1f}"
            details += ")"
            detail_text = small_font.render(details, True, GRAY)
            monitor_screen.blit(detail_text, (500, config_y))
            
            config_y += 20
        
        y_pos += 140
        
        # Statistics Section
        pygame.draw.rect(monitor_screen, LIGHT_GRAY, (10, y_pos, 680, 60), border_radius=5)
        
        stats_title = font.render("Session Statistics", True, BLACK)
        monitor_screen.blit(stats_title, (20, y_pos + 10))
        
        stats_y = y_pos + 35
        stats = [
            f"Focus Time: {self.focus_time:.1f}s",
            f"Relax Time: {self.relax_time:.1f}s", 
            f"Blinks: {self.blinks_detected}",
            f"Noise: {self.noise_level:.1%}"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = small_font.render(stat, True, BLACK)
            monitor_screen.blit(stat_text, (20 + i * 170, stats_y))
        
        # Blink indicator
        if self.blink_detected:
            pygame.draw.circle(monitor_screen, YELLOW, (650, 50), 15)
            blink_text = small_font.render("BLINK!", True, BLACK)
            monitor_screen.blit(blink_text, (625, 70))
        
        # Instructions with enhanced features
        instructions = [
            "🎯 Focus thoughts to hold SPACE key",
            "😌 Relax to press R key", 
            "👁️ Blink to press B key",
            "ESC=Config, C/R=Calibrate, B=Blink sens., T=Test, Q=Quit"
        ]
        
        instr_y = y_pos + 75
        for i, instruction in enumerate(instructions):
            color = DARK_BLUE if i < 3 else GRAY
            size = small_font if i < 3 else small_font
            instr_text = size.render(instruction, True, color)
            monitor_screen.blit(instr_text, (10, instr_y + i * 18))
        
        # Show calibration status
        if self.calibration_mode:
            cal_text = font.render("🎯 CALIBRATING... Follow instructions in console", True, RED)
            monitor_screen.blit(cal_text, (10, 50))
        
        # Show adaptive thresholds if calibrated
        if any(self.calibration_data.values()):
            thresh_y = instr_y + 80
            cal_status = small_font.render("✅ Using calibrated thresholds:", True, GREEN)
            monitor_screen.blit(cal_status, (10, thresh_y))
            
            focus_thresh = small_font.render(f"Focus β: {self.user_thresholds['focus_beta']:.3f}", True, GREEN)
            monitor_screen.blit(focus_thresh, (20, thresh_y + 18))
            
            relax_thresh = small_font.render(f"Relax α: {self.user_thresholds['relax_alpha']:.3f}", True, BLUE)
            monitor_screen.blit(relax_thresh, (150, thresh_y + 18))
        
        pygame.display.flip()
    
    def handle_key_input(self, event):
        """Handle configuration key inputs"""
        if event.key == pygame.K_ESCAPE:
            self.show_config_menu()
        elif event.key == pygame.K_1:
            self.command_config["Focus"]["enabled"] = not self.command_config["Focus"]["enabled"]
        elif event.key == pygame.K_2:
            self.command_config["Relax"]["enabled"] = not self.command_config["Relax"]["enabled"]
        elif event.key == pygame.K_3:
            self.command_config["Blink"]["enabled"] = not self.command_config["Blink"]["enabled"]
        elif event.key == pygame.K_b:
            print("\n=== BLINK CALIBRATION INSTRUCTIONS ===")
            print("1. Make sure you're sitting still")
            print("2. Look straight at the screen")
            print("3. Try to minimize movement")
            print("4. Keep your eyes open and DON'T BLINK")
            print("5. Calibration will take 5 seconds")
            print("\nStarting in 3 seconds...")
            time.sleep(3)
            self.calibrate_blink()
        elif event.key == pygame.K_UP:
            self.adjust_sensitivity(increase=True)
        elif event.key == pygame.K_DOWN:
            self.adjust_sensitivity(increase=False)
    
    def start_calibration(self, state_type, duration=10):
        """Start calibration for a specific mental state"""
        print(f"\n🎯 Starting {duration}s calibration for {state_type}")
        print(f"Please maintain a {state_type.lower()} state now...")
        
        self.calibration_mode = True
        start_time = time.time()
        calibration_samples = []
        
        while time.time() - start_time < duration:
            if hasattr(self, 'alpha_power') and hasattr(self, 'beta_power'):
                calibration_samples.append({
                    'alpha': self.alpha_power,
                    'beta': self.beta_power,
                    'theta': self.theta_power,
                    'timestamp': time.time()
                })
            time.sleep(0.1)
        
        self.calibration_mode = False
        
        if calibration_samples:
            self.calibration_data[state_type.lower()] = calibration_samples
            self.update_user_thresholds()
            print(f"✅ Calibration complete for {state_type}. Collected {len(calibration_samples)} samples.")
        else:
            print(f"❌ No data collected during {state_type} calibration")
    
    def update_user_thresholds(self):
        """Update detection thresholds based on calibration data"""
        try:
            if self.calibration_data["focus"]:
                focus_betas = [s['beta'] for s in self.calibration_data["focus"]]
                self.user_thresholds["focus_beta"] = np.mean(focus_betas) * 0.8  # 80% of mean
                
            if self.calibration_data["relax"]:
                relax_alphas = [s['alpha'] for s in self.calibration_data["relax"]]
                self.user_thresholds["relax_alpha"] = np.mean(relax_alphas) * 0.8
                
            print(f"🔧 Updated thresholds - Focus Beta: {self.user_thresholds['focus_beta']:.3f}, Relax Alpha: {self.user_thresholds['relax_alpha']:.3f}")
        except Exception as e:
            print(f"Error updating thresholds: {e}")
    
    def adaptive_classify_mental_state(self, alpha_avg, beta_avg, theta_avg):
        """Adaptive classification that uses calibrated thresholds when available"""
        # Use calibrated thresholds if available, otherwise fall back to default logic
        focus_threshold = self.user_thresholds.get("focus_beta", 0.5)
        relax_threshold = self.user_thresholds.get("relax_alpha", 0.3)
        
        # Calculate ratios for more robust classification
        alpha_beta_ratio = alpha_avg / (beta_avg + 1e-6)
        total_power = alpha_avg + beta_avg
        
        # Dynamic thresholds based on total power - MATCH BALL GAME
        if total_power < 0.1:  # Very low signal - EXACT THRESHOLD
            return "Neutral", 0.1
        
        # Adaptive Focus detection using calibrated threshold
        if beta_avg > alpha_avg * 1.3 and beta_avg > focus_threshold:
            strength = min(beta_avg / (alpha_avg + beta_avg), 1.0)
            return "Focus", strength
        
        # Adaptive Relax detection using calibrated threshold
        elif alpha_avg > beta_avg * 1.4 and alpha_avg > relax_threshold:
            strength = min(alpha_avg / (alpha_avg + beta_avg), 1.0)
            return "Relax", strength
        
        # Neutral state - EXACT BALL GAME LOGIC
        else:
            return "Neutral", 0.3
        print("\n=== Configuration Menu ===")
        print("1. Toggle Focus command (currently:", "ON" if self.command_config["Focus"]["enabled"] else "OFF", ")")
        print("2. Toggle Relax command (currently:", "ON" if self.command_config["Relax"]["enabled"] else "OFF", ")")
        print("3. Toggle Blink command (currently:", "ON" if self.command_config["Blink"]["enabled"] else "OFF", ")")
        print("Press 1, 2, or 3 to toggle commands")
        print("==========================\n")
    
    def cleanup_keys(self):
        """Release any held keys on exit"""
        try:
            # Always release space key on cleanup
            if self.last_focus_state:
                pyautogui.keyUp(self.command_config["Focus"]["key"])
                self.last_focus_state = False
                print("🔓 Cleanup: Released space key")
            
            # Release any other held keys
            for key, state in self.key_states.items():
                if state:
                    pyautogui.keyUp(key)
                    print(f"🔓 Cleanup: Released {key}")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def run(self):
        """Enhanced main run loop with better error handling"""
        print("🚀 Starting Universal EEG Command Executor v2.0")
        print("Enhanced with:")
        print("  ✅ Ball game exact classification logic")
        print("  ✅ Robust blink detection for FP1/FP2 dry electrodes")
        print("  ✅ User calibration system")
        print("  ✅ Adaptive thresholds")
        print("  ✅ Real-time signal quality monitoring")
        print("\nThis will send keyboard commands to any active application")
        print("Make sure the target application is in focus when you want to control it")
        print("Press ESC in the monitor window for configuration options")
        
        # Start EEG processing thread
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
                clock.tick(30)  # 30 FPS for the monitor
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"Runtime error: {e}")
        
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            self.cleanup_keys()
            self.running = False
            pygame.quit()
            print("👋 EEG Command Executor stopped cleanly")

    def calculate_signal_quality(self, data):
        """Calculate EEG signal quality metrics"""
        try:
            qualities = []
            noises = []
            
            for ch in range(2):  # For each channel (FP1/FP2)
                ch_data = data[:, ch]
                
                # Basic signal checks
                signal_range = np.ptp(ch_data)
                if signal_range < 2:  # Flatline detection
                    return 0.1, 0.9  # Low quality, high noise
                
                # Calculate standard metrics
                signal_mean = np.mean(ch_data)
                signal_std = np.std(ch_data)
                
                # Detect artifacts/outliers
                z_scores = np.abs((ch_data - signal_mean) / (signal_std + 1e-6))
                artifact_ratio = np.mean(z_scores > 4)
                
                # Calculate signal variance quality
                if 10 <= signal_std <= 150:  # Good range for dry electrodes
                    variance_score = 0.8
                else:
                    variance_score = max(0.2, 1.0 - abs(signal_std - 80) / 100)
                
                # Calculate noise level
                noise_level = min(1.0, artifact_ratio + (1.0 if signal_std > 150 else 0.0))
                
                qualities.append(variance_score * (1 - artifact_ratio))
                noises.append(noise_level)
            
            # Average across channels
            quality = np.mean(qualities)
            noise = np.mean(noises)
            
            # Add correlation bonus for nearby electrodes
            try:
                correlation = abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
                if 0.3 <= correlation <= 0.8:  # Expected range for nearby electrodes
                    quality = min(1.0, quality + 0.1)
            except:
                pass
                
            return max(0.1, min(1.0, quality)), min(1.0, max(0.0, noise))
            
        except Exception as e:
            print(f"Signal quality calculation error: {e}")
            return 0.5, 0.5  # Return moderate values on error

if __name__ == "__main__":
    try:
        executor = UniversalEEGCommandExecutor()
        executor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        # Make sure to release keys even on crash
        try:
            pyautogui.keyUp('space')
        except:
            pass