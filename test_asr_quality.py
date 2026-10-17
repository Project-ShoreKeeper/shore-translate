#!/usr/bin/env python3
"""
Shore-Translate: Non-Japanese Speaker ASR Verification & Telemetry Suite
-------------------------------------------------------------------------
Allows developers who do not speak Japanese to objectively verify, benchmark,
and monitor ASR accuracy using mathematical metrics, visual indicators, and
English-based round-trip tests.

Subcommands / Modes:
1. --mode roundtrip
   Synthesizes Japanese speech with exact glossary terms via edge-tts, feeds
   it through VAD -> ASR -> LLM, and calculates Glossary Hit Rate (%), Japanese CER,
   and semantic English match.

2. --mode compare-models --audio <path.wav>
   Runs an audio clip sequentially through multiple Whisper models and outputs a
   clean Markdown comparison table of Latency, VRAM, Logprob, Glossary Recall,
   and Final English Translation.

3. --mode golden-ref --audio <path.wav> --ref-en <subtitles.srt/.txt>
   Compares pipeline output against human-translated English subtitles with
   timestamps and confidence telemetry.
"""

import os
import sys
import time
import json
import asyncio
import argparse
import subprocess
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import soundfile as sf
import numpy as np

# Ensure environment variables
if "XDG_RUNTIME_DIR" not in os.environ:
    uid_runtime = f"/run/user/{os.getuid()}"
    if os.path.exists(uid_runtime):
        os.environ["XDG_RUNTIME_DIR"] = uid_runtime

from config import (
    BASE_DIR,
    SAMPLE_RATE,
    GLOSSARY_PATH,
    LOW_CONF_THRESHOLD,
    HIGH_CONF_THRESHOLD,
    ASR_MODEL_NAME,
)
from vad import VADSegmenter
from asr import WhisperASR, is_valid_japanese
from translate import LiveTranslator, extract_matched_glossary, load_glossary_dict

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False


# ---------------------------------------------------------------------------
# Test Cases for Roundtrip Mode
# ---------------------------------------------------------------------------
ROUNDTRIP_TEST_CASES = [
    {
        "id": 1,
        "english": "Let's merge the pull request into the main branch and deploy the repository.",
        "japanese": "プルリクエストをメインブランチにマージしてリポジトリをデプロイしましょう。",
        "expected_terms": ["プルリクエスト", "ブランチ", "マージ", "リポジトリ", "デプロイ"],
    },
    {
        "id": 2,
        "english": "The database query failed because the server threw an unhandled exception.",
        "japanese": "サーバーが例外をスローしたためデータベースクエリが失敗しました。",
        "expected_terms": ["サーバー", "例外", "データベース", "クエリ"],
    },
    {
        "id": 3,
        "english": "We need to refactor this component to fix the memory leak and ensure type safety.",
        "japanese": "メモリリークを修正し型安全を確保するためにこのコンポーネントをリファクタリングする必要があります。",
        "expected_terms": ["メモリリーク", "型安全", "コンポーネント", "リファクタリング"],
    },
    {
        "id": 4,
        "english": "The asynchronous function returns a promise that resolves with the API response.",
        "japanese": "非同期関数はAPIレスポンスで解決するプロミスを返します。",
        "expected_terms": ["非同期", "関数", "API", "プロミス"],
    },
    {
        "id": 5,
        "english": "Please check the environment variables before pushing changes to the remote branch.",
        "japanese": "リモートブランチに変更をコミットする前に環境変数を確認してください。",
        "expected_terms": ["ブランチ", "コミット", "環境変数"],
    },
]


# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------
def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) using Levenshtein distance:
    CER = (Substitutions + Deletions + Insertions) / len(reference)
    Normalized by removing whitespace and standard punctuation.
    """
    ref = re.sub(r"[\s\u3000、。！？.,!?「」『』]", "", reference)
    hyp = re.sub(r"[\s\u3000、。！？.,!?「」『』]", "", hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    dist = dp[n][m]
    return min(dist / max(len(ref), 1), 1.0)


def compute_glossary_recall(expected_terms: List[str], text: str) -> Tuple[float, List[str], List[str]]:
    """Returns (hit_rate_pct, found_terms, missing_terms)."""
    if not expected_terms:
        return 100.0, [], []
    found = [t for t in expected_terms if t in text]
    missing = [t for t in expected_terms if t not in text]
    hit_rate = (len(found) / len(expected_terms)) * 100.0
    return hit_rate, found, missing


def get_gpu_vram_mb() -> float:
    """Query current GPU VRAM usage via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0


def format_conf_dot(avg_logprob: float) -> str:
    """Visual traffic light confidence indicator."""
    if avg_logprob > HIGH_CONF_THRESHOLD:
        return "🟢"
    elif avg_logprob > LOW_CONF_THRESHOLD:
        return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# TTS Synthesis (Edge-TTS)
# ---------------------------------------------------------------------------
async def _async_synthesize_speech(text: str, output_wav: Path, voice: str) -> bool:
    if not HAS_EDGE_TTS:
        return False
    temp_mp3 = output_wav.with_suffix(".mp3")
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(temp_mp3))

        # Convert mp3 to 16kHz mono float32 wav
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_mp3),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_f32le",
            str(output_wav),
            "-loglevel", "error",
        ]
        res = subprocess.run(cmd, capture_output=True)
        if temp_mp3.exists():
            temp_mp3.unlink()
        return res.returncode == 0 and output_wav.exists()
    except Exception as exc:
        print(f"[tts warning] Synthesis error: {exc}", file=sys.stderr)
        if temp_mp3.exists():
            temp_mp3.unlink()
        return False


def synthesize_speech(text: str, output_wav: Path, voice: str = "ja-JP-NanamiNeural") -> bool:
    """Synchronous wrapper for Edge-TTS synthesis."""
    try:
        return asyncio.run(_async_synthesize_speech(text, output_wav, voice))
    except Exception as exc:
        print(f"[tts warning] Failed to run asyncio synthesis: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Mode 1: Roundtrip Synthetic Verification
# ---------------------------------------------------------------------------
def run_roundtrip_test(voice: str = "ja-JP-NanamiNeural", model_name: Optional[str] = None) -> None:
    print("=" * 72)
    print(" Shore-Translate: Mode 1 - Roundtrip Synthetic TTS & Glossary Verification")
    print("=" * 72)

    active_model = model_name or ASR_MODEL_NAME
    print(f"[config] Active ASR Model : {active_model}")
    print(f"[config] TTS Japanese Voice: {voice}")

    test_dir = BASE_DIR / "debug_asr" / "roundtrip_cache"
    test_dir.mkdir(parents=True, exist_ok=True)

    asr = WhisperASR(model_name=active_model)
    translator = LiveTranslator()

    results = []
    total_expected_terms = 0
    total_found_terms = 0
    total_cer = 0.0

    for tc in ROUNDTRIP_TEST_CASES:
        t_id = tc["id"]
        orig_en = tc["english"]
        ref_ja = tc["japanese"]
        exp_terms = tc["expected_terms"]

        wav_path = test_dir / f"test_{t_id}.wav"

        print(f"\n--- [Test Case {t_id}/5] ---")
        print(f"Original English : \"{orig_en}\"")
        print(f"Target Japanese  : \"{ref_ja}\"")
        print(f"Expected Glossary: {exp_terms}")

        # 1. Synthesize audio via Edge-TTS if not already cached
        if not wav_path.exists():
            print("  -> Synthesizing speech via Edge-TTS...")
            ok = synthesize_speech(ref_ja, wav_path, voice=voice)
            if not ok:
                print(f"  [Error] Failed to synthesize audio for test case {t_id}. Skipping.")
                continue

        audio, sr = sf.read(str(wav_path))
        audio = audio.astype(np.float32)

        # 2. Transcribe via ASR
        t0_asr = time.perf_counter()
        asr_res = asr.transcribe_segment(audio)
        asr_latency = time.perf_counter() - t0_asr

        if not asr_res:
            print("  [Fail] ASR returned empty or filtered output!")
            results.append({
                "id": t_id,
                "orig_en": orig_en,
                "ref_ja": ref_ja,
                "hyp_ja": "[FILTERED/EMPTY]",
                "cer": 1.0,
                "hit_rate": 0.0,
                "conf": "🔴",
                "avg_logprob": -1.0,
                "final_en": "[NONE]",
            })
            continue

        hyp_ja = asr_res[0]
        meta = asr_res[2] if len(asr_res) > 2 else {}
        avg_lp = meta.get("avg_logprob", 0.0)
        dot = format_conf_dot(avg_lp)

        # 3. Calculate CER and Glossary Hit Rate
        cer = compute_cer(ref_ja, hyp_ja)
        hit_rate, found_t, missing_t = compute_glossary_recall(exp_terms, hyp_ja)

        total_expected_terms += len(exp_terms)
        total_found_terms += len(found_t)
        total_cer += cer

        print(f"ASR Output [{dot}] : \"{hyp_ja}\" (logprob: {avg_lp:.2f} | {asr_latency:.2f}s)")
        print(f"CER              : {cer * 100.0:.1f}%")
        print(f"Glossary Hit Rate: {hit_rate:.1f}% (Found: {found_t}, Missing: {missing_t})")

        # 4. Translate back to English via LLM
        t0_trans = time.perf_counter()
        final_en = translator.translate(hyp_ja, is_final=True)
        trans_lat = time.perf_counter() - t0_trans

        print(f"Final English Sub: \"{final_en}\" ({trans_lat:.2f}s)")

        results.append({
            "id": t_id,
            "orig_en": orig_en,
            "ref_ja": ref_ja,
            "hyp_ja": hyp_ja,
            "cer": cer,
            "hit_rate": hit_rate,
            "conf": dot,
            "avg_logprob": avg_lp,
            "final_en": final_en,
        })

    # Summary Table
    print("\n" + "=" * 72)
    print(" ROUNDTRIP BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"| ID | Conf |  CER  | Glossary Hit | Transcribed Japanese             | Final English Subtitle")
    print(f"|---|:---:|:-----:|:------------:|----------------------------------|-----------------------")
    for r in results:
        cer_str = f"{r['cer']*100:.1f}%"
        hit_str = f"{r['hit_rate']:.0f}%"
        ja_trunc = r['hyp_ja'][:28] + ("..." if len(r['hyp_ja']) > 28 else "")
        en_trunc = r['final_en'][:30] + ("..." if len(r['final_en']) > 30 else "")
        print(f"| {r['id']} |  {r['conf']}  | {cer_str:>5} |    {hit_str:>4}    | {ja_trunc:<32} | {en_trunc}")

    avg_cer = (total_cer / len(results)) * 100.0 if results else 0.0
    overall_glossary_pct = (total_found_terms / total_expected_terms * 100.0) if total_expected_terms > 0 else 0.0

    print("-" * 72)
    print(f"Overall Japanese CER           : {avg_cer:.1f}% (Lower is better)")
    print(f"Overall Technical Glossary Hit : {overall_glossary_pct:.1f}% ({total_found_terms}/{total_expected_terms} terms)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Mode 2: A/B Model Benchmark
# ---------------------------------------------------------------------------
def run_compare_models_test(audio_file: str, candidate_models: Optional[List[str]] = None) -> None:
    print("=" * 72)
    print(" Shore-Translate: Mode 2 - CTranslate2 Model A/B Benchmark")
    print("=" * 72)

    if not os.path.exists(audio_file):
        print(f"[Error] Audio file not found: {audio_file}", file=sys.stderr)
        return

    audio, sr = sf.read(audio_file)
    audio = audio.astype(np.float32)
    dur = len(audio) / sr
    print(f"Input Audio: {audio_file} ({dur:.2f}s, {sr} Hz)")

    if not candidate_models:
        candidate_models = [
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            "large-v3",
            "deepdml/faster-whisper-large-v3-turbo-ct2",
        ]

    translator = LiveTranslator()
    glossary = load_glossary_dict()

    comparison_results = []

    for model_name in candidate_models:
        print(f"\n>>> Benchmarking model: {model_name}...")
        vram_before = get_gpu_vram_mb()

        try:
            asr = WhisperASR(model_name=model_name)
        except Exception as exc:
            print(f"  [Skip] Could not load '{model_name}': {exc}")
            continue

        vram_after = get_gpu_vram_mb()
        vram_delta = max(vram_after - vram_before, 0.0)

        # Warm-up inference
        _ = asr.transcribe_segment(audio[: min(len(audio), int(sr * 1.0))])

        # Benchmark run
        t0 = time.perf_counter()
        asr_res = asr.transcribe_segment(audio)
        latency = time.perf_counter() - t0

        if not asr_res:
            print("  [Result] ASR output filtered or empty.")
            comparison_results.append({
                "model": model_name.split("/")[-1],
                "latency_ms": int(latency * 1000),
                "rtf": round(latency / dur, 2),
                "vram_mb": int(vram_after),
                "avg_logprob": -1.0,
                "conf": "🔴",
                "glossary_hit": 0,
                "text": "[FILTERED]",
                "translation": "[NONE]",
            })
            del asr
            continue

        text = asr_res[0]
        meta = asr_res[2] if len(asr_res) > 2 else {}
        avg_lp = meta.get("avg_logprob", 0.0)
        dot = format_conf_dot(avg_lp)

        # Match glossary terms
        matched = extract_matched_glossary(text, glossary)
        glossary_hit = len(matched)

        # Translate to English
        en_trans = translator.translate(text, is_final=True)

        comparison_results.append({
            "model": model_name.split("/")[-1],
            "latency_ms": int(latency * 1000),
            "rtf": round(latency / dur, 2),
            "vram_mb": int(vram_after),
            "avg_logprob": avg_lp,
            "conf": dot,
            "glossary_hit": glossary_hit,
            "text": text,
            "translation": en_trans,
        })

        del asr

    # Print Comparison Markdown Table
    print("\n" + "=" * 72)
    print(" MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 72)
    print("| Model Name | Latency (ms) | RTF | VRAM (MB) | Conf | Logprob | Glossary Hits | English Translation Preview |")
    print("|:-----------|:------------:|:---:|:---------:|:----:|:-------:|:-------------:|:----------------------------|")
    for cr in comparison_results:
        trans_preview = cr["translation"][:35] + ("..." if len(cr["translation"]) > 35 else "")
        print(f"| {cr['model']:<18} | {cr['latency_ms']:>8} ms | {cr['rtf']:>3}x | {cr['vram_mb']:>6} MB |  {cr['conf']}  | {cr['avg_logprob']:>7.2f} | {cr['glossary_hit']:>13} | {trans_preview} |")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Mode 3: Golden Reference Alignment & Comparison
# ---------------------------------------------------------------------------
def parse_subtitles_file(ref_path: Path) -> List[Tuple[float, float, str]]:
    """Parse .srt or .txt file into list of (start_sec, end_sec, text)."""
    items = []
    if not ref_path.exists():
        return items

    content = ref_path.read_text(encoding="utf-8")
    ext = ref_path.suffix.lower()

    if ext == ".srt":
        blocks = re.split(r"\n\s*\n", content.strip())
        for b in blocks:
            lines = [l.strip() for l in b.split("\n") if l.strip()]
            if len(lines) >= 3:
                # 00:00:01,200 --> 00:00:04,500
                m = re.search(r"(\d+):(\d+):(\d+)[,\.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,\.](\d+)", lines[1])
                if m:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
                    t_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                    t_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                    text = " ".join(lines[2:])
                    items.append((t_start, t_end, text))
    else:
        # Plain text file (one line per utterance)
        for idx, line in enumerate(content.splitlines()):
            line = line.strip()
            if line:
                items.append((float(idx), float(idx + 1), line))

    return items


def run_golden_ref_test(audio_file: str, ref_en_file: Optional[str]) -> None:
    print("=" * 72)
    print(" Shore-Translate: Mode 3 - Golden Reference Alignment & Comparison")
    print("=" * 72)

    if not os.path.exists(audio_file):
        print(f"[Error] Audio file not found: {audio_file}", file=sys.stderr)
        return

    ref_items = []
    if ref_en_file:
        ref_items = parse_subtitles_file(Path(ref_en_file))
        print(f"Loaded {len(ref_items)} human English subtitle entries from {ref_en_file}")
    else:
        print("[Notice] No --ref-en file provided. Displaying end-to-end pipeline transcripts and telemetry.")

    audio, sr = sf.read(audio_file)
    audio = audio.astype(np.float32)

    # 1. Segment via VAD
    print("Running Silero VAD segmentation...")
    segmenter = VADSegmenter()
    chunk_size = 480
    segments = []
    for i in range(0, len(audio), chunk_size):
        segments.extend(segmenter.process_samples(audio[i : i + chunk_size]))
    segments.extend(segmenter.flush())

    final_segments = [s for s in segments if (s[3] if len(s) == 4 else True)]
    print(f"Extracted {len(final_segments)} speech segments from audio.")

    # 2. Process through ASR & LLM
    asr = WhisperASR()
    translator = LiveTranslator()

    print("\n" + "=" * 72)
    print(" PIPELINE OUTPUT VS. HUMAN REFERENCE")
    print("=" * 72)

    for idx, seg in enumerate(final_segments, 1):
        if len(seg) == 4:
            epoch_id, seg_audio, seg_dur, is_final = seg
        else:
            seg_audio, seg_dur = seg[:2]
            epoch_id, is_final = 0, True

        asr_res = asr.transcribe_segment(seg_audio)
        if not asr_res:
            continue

        hyp_ja = asr_res[0]
        meta = asr_res[2] if len(asr_res) > 2 else {}
        avg_lp = meta.get("avg_logprob", 0.0)
        dot = format_conf_dot(avg_lp)

        pipe_en = translator.translate(hyp_ja, is_final=True)

        human_ref = "[No alignment provided]"
        if idx - 1 < len(ref_items):
            human_ref = ref_items[idx - 1][2]

        print(f"\n--- Segment {idx} [{dot} logprob: {avg_lp:.2f}] ---")
        print(f"  [ASR Japanese]        : {hyp_ja}")
        print(f"  [Human English Sub]   : {human_ref}")
        print(f"  [shore-translate Sub] : {pipe_en}")

    print("\n" + "=" * 72)
    print(" Golden Reference Evaluation Completed.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Shore-Translate: Non-Japanese Speaker ASR Verification & Telemetry Suite"
    )
    parser.add_argument(
        "--mode",
        choices=["roundtrip", "compare-models", "golden-ref"],
        default="roundtrip",
        help="Evaluation mode to run (default: roundtrip)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="sample_japanese_speech.wav",
        help="Audio file path for compare-models or golden-ref mode",
    )
    parser.add_argument(
        "--ref-en",
        type=str,
        default=None,
        help="Path to reference English subtitles (.srt or .txt) for golden-ref mode",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names for compare-models mode",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="ja-JP-NanamiNeural",
        help="Edge-TTS voice for synthetic audio generation in roundtrip mode",
    )

    args = parser.parse_args()

    if args.mode == "roundtrip":
        run_roundtrip_test(voice=args.voice)
    elif args.mode == "compare-models":
        candidate_models = [m.strip() for m in args.models.split(",")] if args.models else None
        run_compare_models_test(args.audio, candidate_models=candidate_models)
    elif args.mode == "golden-ref":
        run_golden_ref_test(args.audio, args.ref_en)


if __name__ == "__main__":
    main()
