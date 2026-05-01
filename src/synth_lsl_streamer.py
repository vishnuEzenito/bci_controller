#!/usr/bin/env python3
"""
Synthetic EEG LSL streamer — publishes an ORIC-compatible stream for testing
without a physical device.

Stream format matches wirelessdevice_to_lsl.py exactly:
  Name: ORIC  |  Type: EEG  |  8 channels  |  250 Hz  |  float32

Interactive keys (press then Enter):
  f  → focus mode    (high beta, low alpha)
  r  → relax mode    (high alpha, low beta)
  s  → stress mode   (FAA asymmetry: FP1 alpha suppressed, FP2 alpha enhanced)
  n  → neutral mode  (balanced baseline)
  b  → inject blink event
  a  → inject alpha block event
  q  → quit

XDF replay:
  python src/synth_lsl_streamer.py --xdf path/to/file.xdf
  Falls back to synthetic generation if pyxdf is not installed or file missing.
"""

import sys
import argparse
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from synth_eeg import SynthEEGGenerator

# Stream configuration — must match the real ORIC device
N_CHANNELS = 8
SRATE = 250
STREAM_NAME = "ORIC"
STREAM_TYPE = "EEG"
SOURCE_ID = "synth_oric_v2"

CHUNK_SIZE = int(SRATE / 100) or 1  # 10 ms worth of samples


def _keyboard_listener(gen: SynthEEGGenerator, stop_event: threading.Event) -> None:
    """Read single-character commands from stdin (press key then Enter)."""
    state_labels = {'f': 'focus', 'r': 'relax', 's': 'stress', 'n': 'neutral'}
    print("\nStream running. Commands (press key + Enter):")
    print("  f=focus  r=relax  s=stress  n=neutral  b=blink  a=alpha_block  q=quit\n")
    for line in sys.stdin:
        cmd = line.strip().lower()
        if not cmd:
            continue
        if cmd == 'q':
            stop_event.set()
            break
        elif cmd in state_labels:
            gen.set_state(state_labels[cmd])
            print(f"[synth] state → {state_labels[cmd]}")
        elif cmd == 'b':
            gen.inject_event('blink')
            print("[synth] blink injected")
        elif cmd == 'a':
            gen.inject_event('alpha_block')
            print("[synth] alpha_block injected (500 ms alpha suppression)")
        else:
            print(f"[synth] unknown command: {cmd!r}")


def _stream_synthetic(outlet: StreamOutlet, stop_event: threading.Event) -> None:
    gen = SynthEEGGenerator(fs=SRATE, duration=None, channels=N_CHANNELS)

    kbd_thread = threading.Thread(target=_keyboard_listener, args=(gen, stop_event), daemon=True)
    kbd_thread.start()

    print(f"[synth] Streaming synthetic EEG as '{STREAM_NAME}' ({N_CHANNELS} ch @ {SRATE} Hz)")

    while not stop_event.is_set():
        _, samples = gen.next_chunk(CHUNK_SIZE)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        for row in samples:
            outlet.push_sample(row.tolist())
        time.sleep(CHUNK_SIZE / float(SRATE))


def _stream_xdf(xdf_path: str, outlet: StreamOutlet, stop_event: threading.Event) -> None:
    try:
        import pyxdf
    except ImportError:
        print("[synth] pyxdf not installed — falling back to synthetic stream")
        _stream_synthetic(outlet, stop_event)
        return

    print(f"[synth] Loading XDF: {xdf_path}")
    try:
        streams, _ = pyxdf.load_xdf(xdf_path)
    except Exception as e:
        print(f"[synth] Failed to load XDF ({e}) — falling back to synthetic stream")
        _stream_synthetic(outlet, stop_event)
        return

    # Find first EEG stream
    eeg_stream = next(
        (s for s in streams if s['info']['type'][0].upper() == 'EEG'), None
    )
    if eeg_stream is None:
        print("[synth] No EEG stream found in XDF — falling back to synthetic stream")
        _stream_synthetic(outlet, stop_event)
        return

    samples = np.array(eeg_stream['time_series'], dtype=np.float32)  # (n_samples, n_channels)
    n_samples, n_ch = samples.shape
    print(f"[synth] XDF loaded: {n_samples} samples × {n_ch} channels — replaying as ORIC (looping)")

    # Pad or truncate to N_CHANNELS
    if n_ch < N_CHANNELS:
        pad = np.zeros((n_samples, N_CHANNELS - n_ch), dtype=np.float32)
        samples = np.hstack([samples, pad])
    else:
        samples = samples[:, :N_CHANNELS]

    dt = 1.0 / SRATE
    ptr = 0
    while not stop_event.is_set():
        outlet.push_sample(samples[ptr].tolist())
        ptr = (ptr + 1) % n_samples
        time.sleep(dt)


DEMO_SEQUENCE = [
    # (duration_s, state_or_None, event_or_None)
    (4.0,  'neutral',     None),
    (6.0,  'focus',       None),
    (2.0,  None,          'blink'),
    (4.0,  'neutral',     None),
    (6.0,  'relax',       None),
    (4.0,  'neutral',     None),
    (6.0,  'stress',      None),
    (2.0,  None,          'alpha_block'),
    (4.0,  'neutral',     None),
    (2.0,  None,          'alpha_block'),
    (4.0,  'focus',       None),
    (2.0,  None,          'blink'),
    (4.0,  'neutral',     None),
]


def _run_demo(gen: SynthEEGGenerator, stop_event: threading.Event) -> None:
    print("[demo] starting auto-sequence:")
    for dur, state, event in DEMO_SEQUENCE:
        if stop_event.is_set():
            break
        if state:
            gen.set_state(state)
            print(f"[demo] → state: {state}  (for {dur:.0f}s)")
        if event:
            gen.inject_event(event)
            print(f"[demo] ↯ event: {event}")
        time.sleep(dur)
    print("[demo] sequence complete — holding last state")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic ORIC LSL streamer")
    parser.add_argument("--xdf",  metavar="FILE", help="Path to XDF file for replay")
    parser.add_argument("--demo", action="store_true", help="Auto-cycle through states (no keyboard needed)")
    args = parser.parse_args()

    info = StreamInfo(STREAM_NAME, STREAM_TYPE, N_CHANNELS, SRATE, "float32", SOURCE_ID)
    outlet = StreamOutlet(info)
    print(f"[synth] LSL outlet created: {STREAM_NAME}")

    stop_event = threading.Event()

    try:
        if args.xdf:
            _stream_xdf(args.xdf, outlet, stop_event)
        elif args.demo:
            # Start synthetic stream then fire the demo sequence on top
            gen = SynthEEGGenerator(fs=SRATE, duration=None, channels=N_CHANNELS)
            demo_thread = threading.Thread(target=_run_demo, args=(gen, stop_event), daemon=True)
            demo_thread.start()

            print(f"[synth] Demo mode — streaming as '{STREAM_NAME}' ({N_CHANNELS} ch @ {SRATE} Hz)")
            while not stop_event.is_set():
                _, samples = gen.next_chunk(CHUNK_SIZE)
                if samples.ndim == 1:
                    samples = samples.reshape(-1, 1)
                for row in samples:
                    outlet.push_sample(row.tolist())
                time.sleep(CHUNK_SIZE / float(SRATE))
        else:
            _stream_synthetic(outlet, stop_event)
    except KeyboardInterrupt:
        print("\n[synth] Stopped")


if __name__ == "__main__":
    main()
