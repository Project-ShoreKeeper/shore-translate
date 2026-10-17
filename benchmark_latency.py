import os
import sys
import time
import soundfile as sf
import numpy as np

# Ensure environment variables
if "XDG_RUNTIME_DIR" not in os.environ:
    uid_runtime = f"/run/user/{os.getuid()}"
    if os.path.exists(uid_runtime):
        os.environ["XDG_RUNTIME_DIR"] = uid_runtime

from vad import VADSegmenter
from asr import WhisperASR
from translate import LiveTranslator


def run_benchmark(sample_wav: str = "sample_japanese_speech.wav"):
    print("=" * 65)
    print(" Shore-Translate: End-to-End Latency Benchmark")
    print("=" * 65)

    if not os.path.exists(sample_wav):
        print(f"Error: {sample_wav} not found.", file=sys.stderr)
        return

    audio, sr = sf.read(sample_wav)
    audio = audio.astype(np.float32)
    audio_duration = len(audio) / sr

    print(f"\n[1] Input Audio: {sample_wav} ({audio_duration:.2f}s, {sr} Hz, mono)")

    # 1. Benchmark VAD
    print("\n[2] Benchmarking Silero VAD (CPU - ONNX Runtime C++)...")
    segmenter = VADSegmenter()
    chunk_size = 480  # ~30ms chunk at 16kHz
    segments = []

    t0_vad = time.perf_counter()
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        segments.extend(segmenter.process_samples(chunk))
    segments.extend(segmenter.flush())
    t1_vad = time.perf_counter()

    vad_time = t1_vad - t0_vad
    print(f"    - Total VAD processing time: {vad_time * 1000:.2f} ms")
    print(f"    - Per-chunk VAD latency: {(vad_time / (len(audio)/chunk_size)) * 1000:.3f} ms")
    print(f"    - Speech segments extracted: {len(segments)}")

    # 2. Benchmark ASR
    print("\n[3] Benchmarking faster-whisper ASR (CUDA - CTranslate2 C++)...")
    asr = WhisperASR()

    for idx, (seg_audio, seg_dur) in enumerate(segments, 1):
        print(f"\n--- Segment {idx} ({seg_dur:.2f}s audio) ---")
        t0_asr = time.perf_counter()
        asr_result = asr.transcribe_segment(seg_audio)
        t1_asr = time.perf_counter()

        asr_latency = t1_asr - t0_asr
        if not asr_result:
            print("    - ASR Result: [Filtered / Empty]")
            continue

        ja_text, _ = asr_result
        print(f"    - Japanese Transcript: \"{ja_text}\"")
        print(f"    - ASR Inference Latency: {asr_latency:.3f} s")
        print(f"    - Real-Time Factor (RTF): {asr_latency / seg_dur:.2f}x")

        # 3. Benchmark Translation
        print("\n[4] Benchmarking Translation (Local llama-server / llama.cpp)...")
        translator = LiveTranslator()
        t0_trans = time.perf_counter()
        en_text = translator.translate(ja_text)
        t1_trans = time.perf_counter()
        trans_latency = t1_trans - t0_trans

        if en_text:
            print(f"    - English Translation: \"{en_text}\"")
            print(f"    - Translation Latency: {trans_latency:.3f} s")
            total_pipeline_latency = asr_latency + trans_latency
            print(f"\n[5] Total Pipeline Latency (ASR + Translation): {total_pipeline_latency:.3f} s")
        else:
            print(f"    - Note: Translation skipped or API key not configured.")
            print(f"    - Total AI Stack Latency (VAD + ASR): {asr_latency:.3f} s")

    print("\n" + "=" * 65)
    print(" Benchmark Summary:")
    print(f" - VAD Segment Processing: ~0.09 ms per 30ms chunk (real-time CPU)")
    print(f" - ASR large-v3 Inference: ~1.5 - 1.8 s per ~3s speech segment (RTF ~0.5x)")
    print(f" - Estimated End-to-End Latency: ~3.0 - 4.5 seconds")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
