# synth_eeg.py
"""
Synthetic EEG generator supporting both finite and infinite streaming.

API:
- SynthEEGGenerator(fs=250, duration=None).next_chunk(n) -> (t_chunk, samples_chunk)
    - If duration is None, the stream is infinite and time starts at 0s, increasing without bound.
    - If duration is a float (seconds), a finite signal is precomputed; when exhausted, the stream wraps.
- reset() resets the internal pointer/time to start from 0.
- set_state(mode)     — 'neutral' | 'focus' | 'relax' | 'stress'
- inject_event(type)  — 'blink' | 'alpha_block'
"""

import threading
import numpy as np
from numpy.random import default_rng
import scipy.signal as sig

rng = default_rng()

def _pink_noise_spectral(n_samples, fs):
    freqs = np.fft.rfftfreq(n_samples, 1.0/fs)
    filt = np.where(freqs == 0, 0.0, 1.0/np.sqrt(freqs))
    white = rng.normal(size=freqs.shape) + 1j * rng.normal(size=freqs.shape)
    spec = white * filt
    x = np.fft.irfft(spec, n_samples)
    return x / (np.std(x) + 1e-12)

def _sine(fs, t, freq, amplitude=1.0, phase=None):
    if phase is None:
        phase = rng.uniform(0, 2*np.pi)
    return amplitude * np.sin(2*np.pi*freq*t + phase)

def _amp_mod(sig_arr, t, low_fs=0.2, depth=0.4):
    mod = 1.0 + depth * np.sin(2*np.pi*low_fs*t + rng.uniform(0, 2*np.pi))
    return sig_arr * mod

class SynthEEGGenerator:
    def __init__(self, fs=250, duration=None, seed=None, channels=1):
        """
        fs: sampling rate (Hz)
        duration: total duration in seconds for a finite signal; if None, stream is infinite.
        """
        if seed is not None:
            global rng
            rng = np.random.default_rng(seed)
        self.fs = fs
        self.duration = duration
        self.infinite = duration is None
        self.channels = int(channels)
        if self.channels < 1:
            self.channels = 1

        # Per-channel random phases for continuity and spatial variation
        self._ph_delta = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_theta = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_alpha = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_beta  = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_gamma = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_line50 = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_line100 = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_amp_alpha = rng.uniform(0, 2*np.pi, size=self.channels)
        self._ph_amp_beta  = rng.uniform(0, 2*np.pi, size=self.channels)

        self._pink_a = 0.995
        self._pink_b = [1.0]
        self._pink_A = [1.0, -self._pink_a]
        self._pink_zi = None

        # Finite mode
        if not self.infinite:
            self.n_total = int(fs * duration)
            self.t = np.arange(self.n_total) / fs
            self._build_full_signal()
            self.ptr = 0
        else:
            self.sample_idx = 0

        # --- State control (infinite mode only) ---
        self._lock = threading.Lock()
        self._state = 'neutral'          # 'neutral' | 'focus' | 'relax' | 'stress'
        self._blink_sample = -1          # global sample index for pending blink center (-1 = none)
        self._alpha_block_end_sample = -1  # suppress alpha until this global sample index

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def set_state(self, mode):
        """Set the EEG simulation state. mode: 'neutral' | 'focus' | 'relax' | 'stress'"""
        with self._lock:
            self._state = mode

    def inject_event(self, event_type):
        """Inject a transient event. event_type: 'blink' | 'alpha_block'"""
        with self._lock:
            if event_type == 'blink':
                # Schedule blink ~50 ms from now
                self._blink_sample = self.sample_idx + int(0.05 * self.fs)
            elif event_type == 'alpha_block':
                # Suppress alpha for the next 500 ms
                self._alpha_block_end_sample = self.sample_idx + int(0.5 * self.fs)

    # ------------------------------------------------------------------

    def _build_full_signal(self):
        n = self.n_total
        t = self.t

        pink = _pink_noise_spectral(n, self.fs) * 5.0

        delta = _sine(self.fs, t, 2.0, amplitude=12.0)
        theta = _sine(self.fs, t, 6.0, amplitude=6.0)
        alpha = _sine(self.fs, t, 10.0, amplitude=20.0)
        beta  = _sine(self.fs, t, 20.0, amplitude=4.0)
        gamma = _sine(self.fs, t, 40.0, amplitude=1.2)

        alpha = _amp_mod(alpha, t, low_fs=0.12, depth=0.55)
        beta  = _amp_mod(beta, t, low_fs=0.25, depth=0.25)

        line = 1.0 * np.sin(2*np.pi*50.0*t + rng.uniform(0, 2*np.pi))
        line += 0.4 * np.sin(2*np.pi*100.0*t + rng.uniform(0, 2*np.pi))

        blink = np.zeros(n)
        blink_centers = [2.2, 6.8]
        for c in blink_centers:
            width = 0.12
            blink += 80.0 * np.exp(-0.5 * ((t - c) / width)**2)

        emg = np.zeros(n)
        burst_center = 5.5
        burst_width = 0.09
        mask = np.exp(-0.5 * ((t - burst_center)/burst_width)**2) > 0.01
        emg_noise = rng.normal(scale=12.0, size=n) * mask
        emg = sig.lfilter([1, -0.97], [1], emg_noise) * 0.9

        eeg = pink + delta + theta + alpha + beta + gamma + line + blink + emg
        eeg = eeg - np.mean(eeg)
        if self.channels > 1:
            gains = rng.uniform(0.9, 1.1, size=self.channels)
            jitters = rng.normal(scale=0.5, size=(n, self.channels))
            multi = (eeg[:, None] * gains[None, :]) + jitters
            self.full_signal = multi.astype(np.float32)
        else:
            self.full_signal = eeg.astype(np.float32)
        self.full_t = t

    def reset(self):
        if self.infinite:
            self.sample_idx = 0
        else:
            self.ptr = 0

    def next_chunk(self, chunk_size):
        """
        Return the next chunk of length chunk_size (number of samples).
        Returns: (t_chunk, samples_chunk)
        - Infinite mode: time starts at 0 and increases monotonically; state/events applied.
        - Finite mode: precomputed buffer; wraps when the end is reached.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if not self.infinite:
            if self.ptr >= self.n_total:
                self.ptr = 0
            end = min(self.ptr + chunk_size, self.n_total)
            chunk = self.full_signal[self.ptr:end]
            t_chunk = self.full_t[self.ptr:end]
            self.ptr = end
            return t_chunk, chunk

        # --- Infinite procedural generation ---

        # Snapshot state and event info (thread-safe)
        with self._lock:
            state = self._state
            blink_sample = self._blink_sample
            alpha_block_end = self._alpha_block_end_sample

        chunk_start = self.sample_idx  # global index of first sample in this chunk
        idx = np.arange(chunk_size, dtype=np.int64)
        t_chunk = (chunk_start + idx) / float(self.fs)
        self.sample_idx += chunk_size

        t = t_chunk
        ch = self.channels
        t2 = t[:, None]  # (n, 1)

        # Amplitude modulations — depth reduced so inter-channel variation stays small
        # over a 3-second window (avoids false FAA from modulation phase differences).
        amp_alpha = 1.0 + 0.10 * np.sin(2*np.pi*0.12*t2 + self._ph_amp_alpha[None, :])
        amp_beta  = 1.0 + 0.10 * np.sin(2*np.pi*0.25*t2 + self._ph_amp_beta[None, :])

        # Rhythmic components — shape (chunk_size, channels)
        delta = 12.0 * np.sin(2*np.pi*2.0*t2  + self._ph_delta[None, :])
        theta =  6.0 * np.sin(2*np.pi*6.0*t2  + self._ph_theta[None, :])
        alpha = 20.0 * np.sin(2*np.pi*10.0*t2 + self._ph_alpha[None, :]) * amp_alpha
        beta  =  4.0 * np.sin(2*np.pi*20.0*t2 + self._ph_beta[None, :])  * amp_beta
        gamma =  1.2 * np.sin(2*np.pi*40.0*t2 + self._ph_gamma[None, :])

        line = 1.0 * np.sin(2*np.pi*50.0*t2  + self._ph_line50[None, :]) \
             + 0.4 * np.sin(2*np.pi*100.0*t2 + self._ph_line100[None, :])

        white = rng.normal(scale=1.0, size=(chunk_size, ch))
        if self._pink_zi is None:
            self._pink_zi = np.zeros(ch)
        pink = np.zeros_like(white)
        for i in range(ch):
            pink[:, i], self._pink_zi[i] = sig.lfilter(
                self._pink_b, self._pink_A, white[:, i], zi=[self._pink_zi[i]]
            )
        pink *= 0.3

        # --- Apply state-based amplitude scaling ---
        alpha_gain = np.ones(ch)
        beta_gain  = np.ones(ch)

        if state == 'neutral':
            # Balance alpha/beta so neither Focus nor Relax classifier fires
            # Default signal is very alpha-dominant (amplitude 20 vs 4) → would always read as Relax
            alpha_gain[:] = 0.3
            beta_gain[:]  = 1.4
        elif state == 'focus':
            alpha_gain[:] = 0.2
            beta_gain[:]  = 2.5
        elif state == 'relax':
            alpha_gain[:] = 3.0
            beta_gain[:]  = 0.2
        elif state == 'stress' and ch >= 2:
            # Stress: asymmetric alpha — FP1 higher than FP2 (left resting, right active).
            # Average gain = (0.4+0.2)/2 = 0.3, same as neutral, so the state classifier
            # stays Neutral while FAA = ln(FP2α)-ln(FP1α) is clearly negative → high stress.
            alpha_gain[0] = 0.4   # FP1: moderate-high → left less active
            alpha_gain[1] = 0.2   # FP2: low → right more active
            beta_gain[:]  = 1.4   # same as neutral

        alpha = alpha * alpha_gain[None, :]
        beta  = beta  * beta_gain[None, :]

        # --- Apply alpha block event (suppress alpha for duration) ---
        if alpha_block_end > chunk_start:
            global_indices = chunk_start + idx  # shape (chunk_size,)
            in_block = global_indices < alpha_block_end  # bool mask
            if np.any(in_block):
                alpha[in_block, :] *= 0.2

        eeg = pink + delta + theta + alpha + beta + gamma + line
        # No chunk-wise mean subtraction — with chunk_size=2, subtracting the 2-sample
        # mean would cancel most of each sinusoidal component (e.g. 75% of alpha at 10Hz).

        # --- Inject blink transient ---
        # Apply the Gaussian envelope across ALL chunks within ±4σ of the blink center,
        # not just the single 2-sample chunk that contains it.
        if blink_sample >= 0:
            sigma_s = max(int(0.06 * self.fs), 1)  # ~15 samples = 60 ms half-width
            global_idx = chunk_start + idx          # idx already defined above
            dist = global_idx - blink_sample
            if np.any(np.abs(dist) < 4 * sigma_s):
                blink_env = 150.0 * np.exp(-0.5 * (dist / sigma_s) ** 2)
                eeg += blink_env[:, None]
            # Clear once the blink center is well behind this chunk
            if chunk_start > blink_sample + 4 * sigma_s:
                with self._lock:
                    if self._blink_sample == blink_sample:
                        self._blink_sample = -1

        return t_chunk, eeg.astype(np.float32)

    def get_full(self):
        return self.full_t, self.full_signal
