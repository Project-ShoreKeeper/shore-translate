import os
import sys
import time
import json
import queue
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Tuple

from config import (
    LOCAL_LLM_URL,
    TRANSLATION_MODEL,
    TRANSLATION_TIMEOUT_SEC,
    TRANSLATION_MAX_TOKENS,
    GLOSSARY_PATH,
)

# Endpoint constant at top of file
LOCAL_ENDPOINT = LOCAL_LLM_URL  # Default: http://localhost:9090/v1/chat/completions


def load_glossary_text(glossary_path: Path = GLOSSARY_PATH) -> str:
    """Load glossary terms formatted as 'Japanese = English' lines."""
    if not glossary_path.exists():
        return ""
    lines = []
    with open(glossary_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                lines.append(stripped)
    return "\n".join(lines)


SYSTEM_PROMPT_BASE = (
    "You are a live subtitle translator for a Japanese programming/technology livestream. "
    "Translate the Japanese sentence into natural, concise English subtitles. Rules: "
    "(1) Output only the English translation, nothing else. "
    "(2) Use standard English technical terms (e.g. repository, compiler, deploy, pull request, API) and follow the glossary exactly when a term appears. "
    "(3) The Japanese text comes from speech recognition and may contain mistakes, especially katakana loanwords misheard as other words; use the surrounding context to fix obvious errors silently. "
    "(4) Do not add information that is not in the source. "
    "(5) If the input is not meaningful speech, output an empty string."
)


def get_full_system_prompt() -> str:
    glossary_lines = load_glossary_text()
    if glossary_lines:
        return f"{SYSTEM_PROMPT_BASE}\n\nGlossary:\n{glossary_lines}"
    return SYSTEM_PROMPT_BASE


class LiveTranslator:
    """
    Translates Japanese ASR output to English using local llama-server (OpenAI-compatible).
    Maintains a rolling context of the last 2 (Japanese, English) pairs.
    """

    def __init__(self, endpoint: str = LOCAL_ENDPOINT, model_name: str = TRANSLATION_MODEL):
        self.endpoint = endpoint
        self.model_name = model_name
        self.system_prompt = get_full_system_prompt()
        self.context: List[Tuple[str, str]] = []
        print(f"[translate] Local translation backend initialized (endpoint: {self.endpoint})")

    def translate(
        self,
        text: str,
        context: Optional[List[Tuple[str, str]]] = None,
        is_final: bool = True,
    ) -> str:
        """
        Translate Japanese text to English with context of last 2 (JA, EN) pairs.
        On any error, skip that sentence, log one short line, and return ''.
        If is_final=False (draft), translates concisely without committing to context.
        """
        if not text or not text.strip():
            return ""

        if context is None:
            context = self.context

        recent_context = context[-2:]

        # Build message history
        messages = [{"role": "system", "content": self.system_prompt}]
        for ja, en in recent_context:
            if ja and en:
                messages.append({"role": "user", "content": ja})
                messages.append({"role": "assistant", "content": en})
        messages.append({"role": "user", "content": text})

        max_tokens = TRANSLATION_MAX_TOKENS if is_final else 60
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=TRANSLATION_TIMEOUT_SEC) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            choice = resp_data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            translated = (msg.get("content") or "").strip()

            # Clean any leftover surrounding quotes
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1].strip()

            # Update rolling context ONLY on final commits
            if translated and is_final:
                self.context.append((text, translated))
                if len(self.context) > 2:
                    self.context = self.context[-2:]

            return translated

        except urllib.error.URLError as uerr:
            print(f"[translate error] Connection to local LLM failed ({uerr.reason}). Is llama-server running?", file=sys.stderr)
            return ""
        except Exception as exc:
            print(f"[translate error] {type(exc).__name__}: {exc}", file=sys.stderr)
            return ""


def run_translation_worker(
    transcript_queue: queue.Queue,
    subtitle_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread: reads (ja_text, asr_latency, audio_dur, is_final) from transcript_queue,
    calls translate(), and puts (en_text, ja_text, total_latency, is_final) into subtitle_queue.
    Drains older intermediate drafts if newer transcripts are available.
    """
    translator = LiveTranslator()

    while not stop_event.is_set():
        try:
            item = transcript_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Drain obsolete draft transcripts if a newer item is available
        latest_item = item
        while not transcript_queue.empty():
            try:
                peek = transcript_queue.get_nowait()
                latest_item = peek
                if peek[3]:  # If is_final is True, prioritize and stop draining
                    break
            except queue.Empty:
                break

        ja_text, asr_latency, audio_dur, is_final = latest_item
        t0 = time.time()
        en_text = translator.translate(ja_text, is_final=is_final)
        trans_latency = time.time() - t0

        if en_text:
            total_latency = asr_latency + trans_latency
            tag = "FINAL" if is_final else "DRAFT"
            print(f"[TRANSLATE {tag}] ({trans_latency:.2f}s | total: {total_latency:.2f}s) JA: {ja_text}")
            print(f"                 EN: {en_text}")
            subtitle_queue.put((en_text, ja_text, total_latency, is_final))


if __name__ == "__main__":
    translator = LiveTranslator()
    sample_text = "だから非同期処理ってプロミスチケットみたいなの発行してたんですよ。"
    print(f"Testing local translation with: {sample_text}")
    output = translator.translate(sample_text)
    print(f"Result: {output}")
