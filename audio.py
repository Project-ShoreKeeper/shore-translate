import os
import sys
import time
import queue
import threading
from typing import Optional
import numpy as np
import soundfile as sf

# Ensure XDG_RUNTIME_DIR is set for PipeWire / PulseAudio access on Linux
if "XDG_RUNTIME_DIR" not in os.environ:
    uid_runtime = f"/run/user/{os.getuid()}"
    if os.path.exists(uid_runtime):
        os.environ["XDG_RUNTIME_DIR"] = uid_runtime

# soundcard expects sys.argv to have at least two elements for program name identification
while len(sys.argv) < 2:
    sys.argv.append("shore-translate")

import soundcard as sc

from config import SAMPLE_RATE, CHUNK_SAMPLES


def get_loopback_microphone():
    """
    Get the loopback (monitor) microphone corresponding to the default speaker,
    or the first available loopback microphone.
    """
    try:
        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        if mic is not None:
            return mic
    except Exception as exc:
        print(f"[audio] Warning: Could not get loopback from default speaker: {exc}")

    # Fallback: scan all microphones with loopback enabled
    for mic in sc.all_microphones(include_loopback=True):
        if mic.isloopback:
            return mic

    raise RuntimeError("No loopback audio device found to capture system output.")


def capture_loopback(
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    sample_rate: int = SAMPLE_RATE,
    chunk_samples: int = CHUNK_SAMPLES,
) -> None:
    """
    Continuously capture system audio via loopback, convert to mono float32,
    and put chunks into audio_queue.
    """
    mic = get_loopback_microphone()
    print(f"[audio] Capturing system audio from loopback device: {mic.name}")

    with mic.recorder(samplerate=sample_rate, channels=1) as recorder:
        while not stop_event.is_set():
            data = recorder.record(numframes=chunk_samples)
            # data has shape (numframes, channels) or (numframes,)
            if data.ndim > 1:
                # Downmix to mono if multi-channel
                data = data.mean(axis=1)
            chunk = data.flatten().astype(np.float32)
            audio_queue.put(chunk)


def record_to_file(
    output_filename: str = "sample_system_audio_10s.wav",
    duration_sec: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """
    Record `duration_sec` seconds of system loopback audio and save to a .wav file.
    """
    mic = get_loopback_microphone()
    total_frames = int(sample_rate * duration_sec)
    print(f"[audio] Recording {duration_sec:.1f}s of system audio from '{mic.name}'...")
    print(f"[audio] Output target: {output_filename} (Sample rate: {sample_rate} Hz, Mono float32)")

    with mic.recorder(samplerate=sample_rate, channels=1) as recorder:
        data = recorder.record(numframes=total_frames)

    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = data.flatten().astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    sf.write(output_filename, audio, sample_rate, subtype="FLOAT")
    print(f"[audio] Recording completed: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
    print(f"[audio] Peak amplitude: {peak:.4f}, RMS: {rms:.4f}")
    if peak == 0.0:
        print("[audio] Note: Audio is silent (peak=0.0). Start playing a video/audio to capture stream sound.")
    else:
        print(f"[audio] Audio signal detected! Successfully saved to {output_filename}")

    return output_filename


if __name__ == "__main__":
    out_file = "sample_system_audio_10s.wav"
    record_to_file(out_file, duration_sec=10.0)
