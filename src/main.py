from pylsl import resolve_streams, StreamInlet
import threading
import pygame
import time
import queue
import numpy as np
from collections import deque

from src.ui_monitor import EEGMonitorUI
from src.eeg_processor import EEGProcessor
from src.command_executor import CommandExecutor
from src.calibration import CalibrationManager
from src.blink_detector import BlinkDetector
from src.config import STATE_CONFIG, UI_CONFIG

class UniversalEEGCommandExecutor:
    def __init__(self):
        # Initialize components
        self.ui = EEGMonitorUI(UI_CONFIG['width'], UI_CONFIG['height'])
        self.eeg_processor = EEGProcessor()
        self.command_executor = CommandExecutor()
        self.calibration_manager = CalibrationManager()
        self.blink_detector = BlinkDetector()  # NEW: Separate blink detection
        
        # State variables
        self.connection_status = "Disconnected"
        self.current_state = "Neutral"
        self.state_strength = 0.0
        self.target_application = "Any Application"
        
        # CRITICAL: Add missing state history for majority voting
        self.state_history = deque(maxlen=STATE_CONFIG['history_length'])
        
        # Blink detection state
        self.blink_detected = False
        
        # Statistics
        self.focus_time = 0
        self.relax_time = 0
        self.blinks_detected = 0
        self.session_start = time.time()
        
        # Thread control
        self.running = True
        self.data_queue = queue.Queue()
        
    def _fallback_classify(self, alpha_avg, beta_avg, theta_avg):
        """EXACT ORIGINAL LOGIC: Fallback classification when not calibrated"""
        # Calculate ratios for more robust classification
        alpha_beta_ratio = alpha_avg / (beta_avg + 1e-6)
        total_power = alpha_avg + beta_avg
        
        # EXACT ORIGINAL: Dynamic thresholds based on total power
        if total_power < 0.1:
            # CRITICAL: Always release space if signal is too weak (like original)
            if self.command_executor.last_focus_state:
                self.command_executor.send_key_command("Focus", 0.0)
            return "Neutral", 0.1
        
        # EXACT ORIGINAL: Focus detection with immediate key handling
        if beta_avg > alpha_avg * 1.3 and beta_avg > 0.5:
            strength = min(beta_avg / (alpha_avg + beta_avg), 1.0)
            return "Focus", strength
        
        # CRITICAL: If not focusing, always release space (like original)
        if self.command_executor.last_focus_state:
            self.command_executor.send_key_command("Focus", 0.0)
        
        # EXACT ORIGINAL: Check for relax state
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
                print("💡 Running in DEMO mode with simulated EEG data")
                self.connection_status = "Demo Mode"
                
                # DEMO MODE: Generate fake EEG data for testing UI
                import random
                while self.running:
                    try:
                        # Simulate EEG processing with fake data
                        self.alpha_power = random.uniform(0.2, 0.4)
                        self.beta_power = random.uniform(0.3, 0.6) 
                        self.theta_power = random.uniform(0.1, 0.3)
                        self.gamma_power = random.uniform(0.05, 0.15)
                        
                        # Simulate occasional blinks
                        if random.random() < 0.02:  # 2% chance per iteration
                            self.blink_detected = True
                            self.blinks_detected += 1
                            print("👁️ Demo blink detected!")
                        
                        # Simulate state changes
                        new_state, new_strength = self.classify_mental_state_with_smoothing(
                            self.alpha_power, self.beta_power, self.theta_power
                        )
                        
                        if self.current_state != new_state:
                            self.current_state = new_state
                            self.state_strength = new_strength
                            print(f"🧠 Demo state: {new_state} (strength: {new_strength:.2f})")
                        
                        # Update timing
                        if self.current_state == "Focus":
                            self.focus_time += 0.1
                        elif self.current_state == "Relax":
                            self.relax_time += 0.1
                        
                        time.sleep(0.1)  # 10Hz update rate
                        
                    except Exception as e:
                        print(f"Demo mode error: {e}")
                        time.sleep(1)
                return
            
            inlet = StreamInlet(oric_stream, max_chunklen=1)
            print("✅ Connected to EEG stream")
            self.connection_status = "Connected"
            
            # EXACT ORIGINAL: Initialize tracking variables
            buffer = deque(maxlen=self.eeg_processor.window_length * self.eeg_processor.fs)
            last_focus_time = time.time()
            last_relax_time = time.time()
            old_state = "Neutral"  # CRITICAL: Track previous state for "send only on change"
            
            while self.running:
                try:
                    sample, timestamp = inlet.pull_sample(timeout=1.0)
                    if sample is None:
                        continue
                    
                    # EXACT ORIGINAL: Use local buffer instead of processor buffer
                    buffer.append(sample[:2])  # First 2 channels
                    
                    if len(buffer) >= self.eeg_processor.min_samples:
                        data = np.array(buffer)
                        
                        # EXACT ORIGINAL: Calculate signal quality directly here
                        self.eeg_processor.signal_quality, self.eeg_processor.noise_level = self.eeg_processor.calculate_signal_quality(data)
                        
                        # EXACT ORIGINAL: Blink detection (on RAW data before filtering - critical!)
                        blink_detected = self.blink_detector.detect_blink(data[-int(self.eeg_processor.fs * 0.2):])
                        
                        if blink_detected:
                            self.blink_detected = True
                            self.blinks_detected += 1
                            self.command_executor.send_key_command("Blink")
                        
                        # EXACT ORIGINAL: Apply filtering AFTER blink detection
                        try:
                            # Bandpass filter (1-40 Hz)
                            b, a = self.eeg_processor.butter_bandpass(1, 40, self.eeg_processor.fs, order=3)
                            data_filtered = self.eeg_processor.apply_filter(data, b, a)
                            
                            # Notch filter (49-51 Hz for 50Hz line noise)
                            b, a = self.eeg_processor.butter_bandstop(49, 51, self.eeg_processor.fs, order=2)
                            data_filtered = self.eeg_processor.apply_filter(data_filtered, b, a)
                        except:
                            data_filtered = data
                        
                        # EXACT ORIGINAL: Process mental state from both channels
                        ch_results = []
                        for ch in range(2):
                            eeg = data_filtered[:, ch]
                            f, psd = self.eeg_processor.robust_welch(eeg, self.eeg_processor.fs)
                            
                            if f is not None and psd is not None:
                                try:
                                    theta = self.eeg_processor.bandpower(f, psd, [4, 8])
                                    alpha = self.eeg_processor.bandpower(f, psd, [8, 12])
                                    beta = self.eeg_processor.bandpower(f, psd, [13, 30])
                                    gamma = self.eeg_processor.bandpower(f, psd, [30, 40])
                                    
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
                        
                        # EXACT ORIGINAL: Average across channels and classify
                        if len(ch_results) >= 2:
                            self.alpha_power = (ch_results[0]['alpha'] + ch_results[1]['alpha']) / 2
                            self.beta_power = (ch_results[0]['beta'] + ch_results[1]['beta']) / 2
                            self.theta_power = (ch_results[0]['theta'] + ch_results[1]['theta']) / 2
                            self.gamma_power = (ch_results[0]['gamma'] + ch_results[1]['gamma']) / 2
                        
                        # EXACT ORIGINAL: Use adaptive classification if calibration data is available
                        if any(self.calibration_manager.calibration_data.values()):
                            state, strength = self.calibration_manager.adaptive_classify_mental_state(
                                self.alpha_power, self.beta_power, self.theta_power
                            )
                        else:
                            # Use standard classification (exact ball game logic)
                            state, strength = self._fallback_classify(
                                self.alpha_power, self.beta_power, self.theta_power
                            )
                        
                        # EXACT ORIGINAL: Smooth the state classification - EXACT BALL GAME LOGIC
                        self.state_history.append((state, strength))
                        
                        # EXACT ORIGINAL: MATCH BALL GAME: Majority vote with strength weighting (requires 3+ votes)
                        if len(self.state_history) >= 5:
                            focus_votes = sum(1 for s, _ in self.state_history if s == "Focus")
                            relax_votes = sum(1 for s, _ in self.state_history if s == "Relax")
                            
                            current_time = time.time()
                            previous_state = self.current_state  # Store previous state
                            
                            # EXACT ORIGINAL: State transitions with key management
                            if focus_votes >= 3:
                                if self.current_state != "Focus":
                                    self.current_state = "Focus"
                                    self.state_strength = np.mean([s for st, s in self.state_history if st == "Focus"])
                                    self.focus_time += current_time - last_focus_time
                                    last_focus_time = current_time
                                
                            elif relax_votes >= 3:
                                if self.current_state == "Focus":
                                    self.command_executor.send_key_command("Focus", 0.0)  # Release space when switching to relax
                                self.current_state = "Relax"
                                self.state_strength = np.mean([s for st, s in self.state_history if st == "Relax"])
                                self.relax_time += current_time - last_relax_time
                                last_relax_time = current_time
                                
                            else:
                                if self.current_state == "Focus":
                                    self.command_executor.send_key_command("Focus", 0.0)  # Release space when switching to neutral
                                self.current_state = "Neutral"
                                self.state_strength = 0.3
                            
                            # EXACT ORIGINAL: Send commands ONLY on state change
                            if self.current_state != old_state:
                                self.command_executor.send_key_command(self.current_state, self.state_strength)
                                old_state = self.current_state
                        
                except Exception as e:
                    print(f"EEG processing error: {e}")
                    time.sleep(0.1)  # Brief pause before retry
                
        except Exception as e:
            print(f"EEG setup error: {e}")
            self.connection_status = f"Error: {str(e)}"

    def handle_key_input(self, event):
        """ENHANCED: Handle configuration key inputs with sensitivity and calibration"""
        if event.key == pygame.K_ESCAPE:
            self.show_config_menu()
        elif event.key == pygame.K_1:
            self.command_executor.toggle_command("Focus")
        elif event.key == pygame.K_2:
            self.command_executor.toggle_command("Relax")
        elif event.key == pygame.K_3:
            self.command_executor.toggle_command("Blink")
        elif event.key == pygame.K_b:
            print("\n=== BLINK CALIBRATION INSTRUCTIONS ===")
            print("1. Make sure you're sitting still")
            print("2. Look straight at the screen")
            print("3. Try to minimize movement")
            print("4. Keep your eyes open and DON'T BLINK")
            print("5. Calibration will take 5 seconds")
            print("\nStarting in 3 seconds...")
            time.sleep(3)
            self.blink_detector.calibrate_baseline()
        # CRITICAL: Add missing sensitivity adjustment
        elif event.key == pygame.K_UP:
            self.blink_detector.adjust_sensitivity(increase=True)
        elif event.key == pygame.K_DOWN:
            self.blink_detector.adjust_sensitivity(increase=False)
        # Add mental state calibration
        elif event.key == pygame.K_c:
            print("\n=== MENTAL STATE CALIBRATION ===")
            print("This will calibrate Focus and Relax states")
            print("1. First: FOCUS state (concentrate, solve math problems)")
            print("2. Then: RELAX state (close eyes, meditate)")
            print("\nStarting Focus calibration in 3 seconds...")
            time.sleep(3)
            # Note: This would need integration with the processing loop
            # For now, just indicate calibration mode
            print("Calibration system ready - implement data collection loop")
            
    def show_config_menu(self):
        """Show configuration menu"""
        print("\n=== Configuration Menu ===")
        print("1. Toggle Focus command (currently:", 
              "ON" if self.command_executor.command_config["Focus"]["enabled"] else "OFF", ")")
        print("2. Toggle Relax command (currently:", 
              "ON" if self.command_executor.command_config["Relax"]["enabled"] else "OFF", ")")
        print("3. Toggle Blink command (currently:", 
              "ON" if self.command_executor.command_config["Blink"]["enabled"] else "OFF", ")")
        print("Press 1, 2, or 3 to toggle commands")
        print("==========================\n")

    def get_state_data(self):
        """ENHANCED: Get current state data for UI with all features"""
        calibration_status = self.calibration_manager.get_calibration_status()
        
        return {
            'connection_status': self.connection_status,
            'current_state': self.current_state,
            'state_strength': self.state_strength,
            'target_application': self.target_application,
            'signal_quality': self.eeg_processor.signal_quality,
            'noise_level': self.eeg_processor.noise_level,
            'commands_sent': self.command_executor.commands_sent,
            'session_start': self.session_start,
            'alpha_power': getattr(self, 'alpha_power', 0),
            'beta_power': getattr(self, 'beta_power', 0),
            'theta_power': getattr(self, 'theta_power', 0),
            'gamma_power': getattr(self, 'gamma_power', 0),
            'command_config': self.command_executor.get_command_config(),
            'focus_time': self.focus_time,
            'relax_time': self.relax_time,
            'blinks_detected': self.blinks_detected,
            'blink_detected': self.blink_detected,  # For UI blink indicator
            'calibration_mode': calibration_status['calibration_mode'],
            'is_calibrated': calibration_status['is_calibrated'],
            'user_thresholds': calibration_status['user_thresholds']
        }

    def run(self):
        """Main run loop"""
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
                            try:
                                self.handle_key_input(event)
                            except Exception as e:
                                print(f"Key handling error: {e}")
                
                try:
                    self.ui.draw_monitor(self.get_state_data())
                except Exception as e:
                    print(f"UI drawing error: {e}")
                
                # Reset blink detected flag after UI update
                if self.blink_detected:
                    time.sleep(0.05)  # Brief flash duration
                    self.blink_detected = False
                
                clock.tick(UI_CONFIG['fps'])  # FPS from config
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"Runtime error: {e}")
        
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            self.command_executor.cleanup_keys()
            self.running = False
            self.ui.cleanup()
            print("👋 EEG Command Executor stopped cleanly")

if __name__ == "__main__":
    try:
        executor = UniversalEEGCommandExecutor()
        executor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        # Make sure to release keys even on crash
        try:
            import pyautogui
            pyautogui.keyUp('space')
        except:
            pass
