import re
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel

from config import (
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
)


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
    """

    def __init__(self, model_name: str = ASR_MODEL_NAME):
        # Strict CUDA check as required: exit with clear error if unavailable
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
        self.hotwords = " ".join(self.ja_terms) if self.ja_terms else None
        print(f"[asr] Loaded {len(self.ja_terms)} glossary terms for initial_prompt/hotwords.")

    def transcribe_segment(self, audio_data: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Transcribe a single 16kHz float32 audio segment.
        Returns (text, latency_sec) if valid, or None if filtered/empty.
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

            # Aggregate segment texts and filter hallucinations
            valid_texts = []
            for seg in segments:
                text = seg.text.strip()
                # Filter criteria from specifications:
                # 1. Empty or shorter than 2 characters
                if len(text) < ASR_MIN_CHAR_LENGTH:
                    continue
                # 2. High no_speech_prob (> 0.6)
                if seg.no_speech_prob > ASR_MAX_NO_SPEECH_PROB:
                    continue
                # 3. Low avg_logprob (< -1.0)
                if seg.avg_logprob < ASR_MIN_AVG_LOGPROB:
                    continue
                # 4. Repetitive hallucination patterns
                if is_repetitive(text):
                    continue

                valid_texts.append(text)

            latency = time.time() - t_start
            full_text = " ".join(valid_texts).strip()

            if len(full_text) < ASR_MIN_CHAR_LENGTH or is_repetitive(full_text):
                return None
            if full_text in KNOWN_HALLUCINATIONS:
                return None
            if is_glossary_echo(full_text, self.ja_terms):
                return None

            return full_text, latency

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
    and puts (text, latency_sec, audio_duration_sec, is_final) into transcript_queue.
    Drains older intermediate drafts if newer speech chunks are waiting.
    """
    asr = WhisperASR()

    while not stop_event.is_set():
        try:
            item = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Drain obsolete draft items if newer items are available
        latest_item = item
        while not speech_queue.empty():
            try:
                peek = speech_queue.get_nowait()
                latest_item = peek
                if peek[2]:  # If is_final is True, prioritize and stop draining
                    break
            except queue.Empty:
                break

        audio_data, audio_duration, is_final = latest_item
        result = asr.transcribe_segment(audio_data)
        if result is not None:
            text, latency = result
            tag = "FINAL" if is_final else "DRAFT"
            print(f"[ASR {tag}] ({audio_duration:.2f}s audio | {latency:.2f}s latency): {text}")
            transcript_queue.put((text, latency, audio_duration, is_final))
