import re
import os
import sys
import time
import json
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import ctranslate2
from faster_whisper import WhisperModel

from config import (
    SAMPLE_RATE,
    ASR_MODEL_NAME,
    ASR_DEVICE,
    ASR_COMPUTE_TYPE,
    ASR_LANGUAGE,
    ASR_BEAM_SIZE,
    ASR_TEMPERATURE,
    ASR_CONDITION_ON_PREVIOUS_TEXT,
    ASR_VAD_FILTER,
    ASR_MIN_CHAR_LENGTH,
    ASR_MAX_NO_SPEECH_PROB,
    ASR_MIN_AVG_LOGPROB,
    GLOSSARY_PATH,
    ENABLE_ASR_TELEMETRY,
    DEBUG_SAVE_LOW_CONF_AUDIO,
    DEBUG_ASR_DIR,
    LOW_CONF_THRESHOLD,
    HIGH_CONF_THRESHOLD,
)

# Japanese script guard regexes
HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")


def is_valid_japanese(text: str) -> bool:
    """
    Japanese Script Guard:
    1. Verifies that output contains Hiragana or Katakana.
    2. Rejects text containing Korean Hangul.
    3. Rejects pure Latin or pure Chinese characters caused by Whisper language drift.
    """
    if not text or not text.strip():
        return False
    if HANGUL_RE.search(text):
        return False
    # Must contain at least one Hiragana or Katakana character
    if not (HIRAGANA_RE.search(text) or KATAKANA_RE.search(text)):
        return False
    return True


def dump_low_conf_audio(
    audio_data: np.ndarray,
    text: str,
    latency: float,
    meta: dict,
) -> None:
    """Save low-confidence audio chunks as .wav and .json for offline diagnostic inspection."""
    try:
        DEBUG_ASR_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        avg_lp = meta.get("avg_logprob", 0.0)
        lp_str = f"{avg_lp:+.2f}".replace(".", "p").replace("-", "m").replace("+", "p")
        base_name = f"chunk_{ts_str}_{ms:03d}_{lp_str}"
        wav_path = DEBUG_ASR_DIR / f"{base_name}.wav"
        json_path = DEBUG_ASR_DIR / f"{base_name}.json"

        sf.write(str(wav_path), audio_data, SAMPLE_RATE, subtype="FLOAT")
        payload = {
            "text": text,
            "latency_sec": latency,
            "audio_duration_sec": len(audio_data) / SAMPLE_RATE,
            "timestamp": time.time(),
            "telemetry": meta,
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        print(f"[asr telemetry] Low-confidence audio dumped to {wav_path} (logprob: {avg_lp:.3f})")
    except Exception as exc:
        print(f"[asr telemetry warning] Failed to dump debug audio: {exc}", file=sys.stderr)


def load_glossary_terms(glossary_path: Path = GLOSSARY_PATH) -> list[str]:
    """Read Japanese terms from glossary.txt."""
    terms = []
    if not glossary_path.exists():
        return terms
    with open(glossary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                ja_part = line.split("=")[0].strip()
                if ja_part:
                    terms.append(ja_part)
    return terms


def build_initial_prompt(terms: list[str]) -> str:
    """Build a natural Japanese prompt setting the programming domain context (< 200 tokens)."""
    return "これはIT・プログラミングやソフトウェア開発に関する技術的な解説配信です。"


KNOWN_HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "ご視聴いただきありがとうございました",
    "ご視聴いただきありがとうございました。",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録よろしくお願いします。",
    "高評価よろしくお願いします",
    "最後までご視聴いただき",
    "最後までご視聴ありがとうございました",
}


def is_glossary_echo(text: str, terms: list[str]) -> bool:
    """Detect when Whisper hallucinates by dumping a comma-separated dictionary list."""
    matched = [t for t in terms if t in text]
    if len(matched) >= 3:
        cleaned = text
        for t in matched:
            cleaned = cleaned.replace(t, "")
        cleaned = re.sub(r"[、, 。\s・\n]", "", cleaned)
        if len(cleaned) <= 3:
            return True
    return False


def is_repetitive(text: str) -> bool:
    """Detect repetitive hallucination patterns in transcribed text."""
    # Pattern 1: Exact substring repetitions (e.g., 'ありがとうありがとうありがとう')
    if re.search(r"(.{2,20})\1{2,}", text):
        return True

    # Pattern 2: Word / token frequency repetition
    words = re.findall(r"[\w]+", text)
    if len(words) >= 4:
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        max_count = max(counts.values())
        if max_count >= 3 and (max_count / len(words)) >= 0.25:
            return True

    # Pattern 3: Very low character diversity for longer phrases
    if len(text) >= 15:
        unique_ratio = len(set(text)) / len(text)
        if unique_ratio < 0.25:
            return True

    return False


class WhisperASR:
    """
    ASR transcription engine powered by faster-whisper (CTranslate2 C++/CUDA backend).
    Features:
    - Japanese Script Guard (Hiragana/Katakana validation, Hangul/drift rejection)
    - Full confidence telemetry (avg_logprob, no_speech_prob, compression_ratio, lang_prob)
    - Low-confidence diagnostic audio dumper
    """

    def __init__(self, model_name: str = ASR_MODEL_NAME):
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count == 0 or ASR_DEVICE != "cuda":
            print(
                "[ASR ERROR] NVIDIA CUDA GPU is required for ASR but not available! "
                f"CTranslate2 reports {cuda_count} CUDA devices.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[asr] Loading Whisper model '{model_name}' on CUDA ({ASR_COMPUTE_TYPE})...")
        t0 = time.time()
        self.model = WhisperModel(
            model_name,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        )
        print(f"[asr] Model loaded in {time.time() - t0:.2f}s.")

        self.ja_terms = load_glossary_terms()
        self.initial_prompt = build_initial_prompt(self.ja_terms)
        # Avoid hotwords buffer overflow on distilled models (e.g. Kotoba) where initial_prompt suffices
        if "kotoba" in model_name.lower():
            self.hotwords = None
        else:
            self.hotwords = " ".join(self.ja_terms[:15]) if self.ja_terms else None
        print(f"[asr] Loaded {len(self.ja_terms)} glossary terms (initial_prompt active, hotwords={'enabled' if self.hotwords else 'disabled'}).")

    def transcribe_segment(self, audio_data: np.ndarray) -> Optional[Tuple[str, float, dict]]:
        """
        Transcribe a single 16kHz float32 audio segment.
        Returns (text, latency_sec, telemetry_dict) if valid, or None if filtered/empty.
        """
        t_start = time.time()
        audio_data = np.asarray(audio_data, dtype=np.float32)
        try:
            kwargs = {
                "language": ASR_LANGUAGE,
                "beam_size": ASR_BEAM_SIZE,
                "temperature": ASR_TEMPERATURE,
                "condition_on_previous_text": ASR_CONDITION_ON_PREVIOUS_TEXT,
                "vad_filter": ASR_VAD_FILTER,
                "initial_prompt": self.initial_prompt if self.initial_prompt else None,
            }
            if self.hotwords:
                kwargs["hotwords"] = self.hotwords

            segments, info = self.model.transcribe(audio_data, **kwargs)

            # Aggregate segment texts and compute weighted confidence metrics
            valid_texts = []
            total_tokens = 0
            total_logprob = 0.0
            total_no_speech = 0.0
            total_compression = 0.0
            seg_count = 0

            for seg in segments:
                seg_count += 1
                total_no_speech += getattr(seg, "no_speech_prob", 0.0)
                total_compression += getattr(seg, "compression_ratio", 1.0)

                seg_tokens = len(seg.tokens) if hasattr(seg, "tokens") and seg.tokens else max(len(seg.text), 1)
                total_logprob += seg.avg_logprob * seg_tokens
                total_tokens += seg_tokens

                text = seg.text.strip()
                if len(text) < ASR_MIN_CHAR_LENGTH:
                    continue
                if seg.no_speech_prob > ASR_MAX_NO_SPEECH_PROB:
                    continue
                if seg.avg_logprob < ASR_MIN_AVG_LOGPROB:
                    continue
                if is_repetitive(text):
                    continue

                valid_texts.append(text)

            latency = time.time() - t_start
            full_text = " ".join(valid_texts).strip()

            avg_logprob = (total_logprob / total_tokens) if total_tokens > 0 else -1.0
            mean_no_speech = (total_no_speech / seg_count) if seg_count > 0 else 0.0
            mean_compression = (total_compression / seg_count) if seg_count > 0 else 1.0
            lang = getattr(info, "language", "ja")
            lang_prob = getattr(info, "language_probability", 1.0)

            meta = {
                "avg_logprob": round(float(avg_logprob), 3),
                "no_speech_prob": round(float(mean_no_speech), 3),
                "compression_ratio": round(float(mean_compression), 3),
                "lang": lang,
                "lang_prob": round(float(lang_prob), 3),
            }

            # Filter hallucination & repetitive patterns
            if len(full_text) < ASR_MIN_CHAR_LENGTH or is_repetitive(full_text):
                return None
            if full_text in KNOWN_HALLUCINATIONS:
                return None
            if is_glossary_echo(full_text, self.ja_terms):
                return None

            # Japanese Script Guard (reject Hangul, pure Latin, or pure Chinese)
            if not is_valid_japanese(full_text):
                return None

            # Low-confidence audio dump
            if DEBUG_SAVE_LOW_CONF_AUDIO and avg_logprob < LOW_CONF_THRESHOLD:
                dump_low_conf_audio(audio_data, full_text, latency, meta)

            return full_text, latency, meta

        except Exception as exc:
            print(f"[asr] Transcription error: {exc}", file=sys.stderr)
            return None


def run_asr_worker(
    speech_queue: queue.Queue,
    transcript_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread: reads speech segments from speech_queue, runs ASR,
    and puts (epoch_id, text, latency_sec, audio_duration_sec, is_final, meta) into transcript_queue.
    Features:
    - Preemption / Drop Stale Drafts
    - Telemetry logging with traffic light indicator (🟢/🟡/🔴)
    """
    asr = WhisperASR()
    committed_epoch = -1

    while not stop_event.is_set():
        try:
            item = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Drain and inspect any additional pending items
        pending_items = [item]
        while not speech_queue.empty():
            try:
                pending_items.append(speech_queue.get_nowait())
            except queue.Empty:
                break

        # Check if there is any final commit among pending items
        final_item = None
        for it in reversed(pending_items):
            is_fin = it[3] if len(it) == 4 else it[2]
            if is_fin:
                final_item = it
                break

        if final_item is not None:
            chosen_item = final_item
        else:
            chosen_item = pending_items[-1]

        if len(chosen_item) == 4:
            epoch_id, audio_data, audio_duration, is_final = chosen_item
        else:
            epoch_id = 0
            audio_data, audio_duration, is_final = chosen_item

        # Discard stale draft
        if not is_final and epoch_id <= committed_epoch:
            continue

        result = asr.transcribe_segment(audio_data)
        if result is not None:
            text = result[0]
            latency = result[1]
            meta = result[2] if len(result) > 2 else {}

            # If this was a draft, check if a Final item arrived while transcribing
            if not is_final and not speech_queue.empty():
                has_final = False
                temp_items = []
                while not speech_queue.empty():
                    try:
                        temp = speech_queue.get_nowait()
                        temp_items.append(temp)
                        if (temp[3] if len(temp) == 4 else temp[2]):
                            has_final = True
                    except queue.Empty:
                        break
                for t in temp_items:
                    speech_queue.put(t)

                if has_final:
                    continue

            tag = "FINAL" if is_final else "DRAFT"
            avg_lp = meta.get("avg_logprob", 0.0)
            if ENABLE_ASR_TELEMETRY:
                dot = "🟢" if avg_lp > HIGH_CONF_THRESHOLD else ("🟡" if avg_lp > LOW_CONF_THRESHOLD else "🔴")
                print(f"[ASR {tag} {dot}] (epoch {epoch_id} | logprob: {avg_lp:.2f} | no_speech: {meta.get('no_speech_prob', 0):.2f} | {audio_duration:.2f}s audio | {latency:.2f}s latency): {text}")
            else:
                print(f"[ASR {tag}] (epoch {epoch_id} | {audio_duration:.2f}s audio | {latency:.2f}s latency): {text}")

            transcript_queue.put((epoch_id, text, latency, audio_duration, is_final, meta))

            if is_final:
                committed_epoch = max(committed_epoch, epoch_id)
