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
    TRANSLATION_TIMEOUT_DRAFT_SEC,
    TRANSLATION_MAX_TOKENS,
    TRANSLATION_MAX_TOKENS_DRAFT,
    GLOSSARY_PATH,
)

LOCAL_ENDPOINT = LOCAL_LLM_URL

# Cải tiến 2: System prompt cố định 100% để tái sử dụng toàn bộ Prefix KV Cache trong llama-server
STATIC_SYSTEM_PROMPT = (
    "You are a live subtitle translator for a Japanese programming and technology livestream. "
    "Translate the Japanese speech into natural, concise English subtitles.\n"
    "Rules:\n"
    "1. Output ONLY the English translation, without explanation, notes, or quotation marks.\n"
    "2. If glossary term definitions are provided, adhere to them strictly.\n"
    "3. The Japanese speech comes from real-time ASR and may contain minor misrecognitions; resolve them naturally using context.\n"
    "4. Do not invent facts not present in the input speech.\n"
    "5. If the input is unintelligible noise, output an empty string."
)


def load_glossary_dict(glossary_path: Path = GLOSSARY_PATH) -> dict[str, str]:
    """
    Load glossary terms formatted as 'Japanese = English' into a dictionary.
    Terms are sorted by length descending so longer compound phrases match first.
    """
    if not glossary_path.exists():
        return {}
    glossary = {}
    with open(glossary_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                parts = stripped.split("=", 1)
                ja = parts[0].strip()
                en = parts[1].strip()
                if ja and en:
                    glossary[ja] = en
    # Sort by key length descending
    return dict(sorted(glossary.items(), key=lambda kv: len(kv[0]), reverse=True))


def extract_matched_glossary(text: str, glossary_dict: dict[str, str]) -> list[tuple[str, str]]:
    """
    Cải tiến 3: Dynamic Glossary Matching.
    So khớp cực nhanh (<0.05ms) tìm các thuật ngữ tiếng Nhật thực sự xuất hiện trong câu nói.
    """
    if not text or not glossary_dict:
        return []
    matched = []
    for ja, en in glossary_dict.items():
        if ja in text:
            matched.append((ja, en))
    return matched


class LiveTranslator:
    """
    Translates Japanese ASR output to English using local llama-server (OpenAI-compatible).
    Features:
    - 100% Static System Prompt for Prefix KV Cache reuse (--cache-reuse in llama.cpp)
    - Dynamic per-turn glossary injection in User turn
    - Rolling context of the last 2 (JA, EN) conversational pairs
    - Fast timeout and reduced token budget for intermediate drafts
    """

    def __init__(self, endpoint: str = LOCAL_ENDPOINT, model_name: str = TRANSLATION_MODEL):
        self.endpoint = endpoint
        self.model_name = model_name
        self.system_prompt = STATIC_SYSTEM_PROMPT
        self.glossary_dict = load_glossary_dict()
        self.context: List[Tuple[str, str]] = []
        print(f"[translate] Local translation backend initialized (endpoint: {self.endpoint})")
        print(f"[translate] Loaded {len(self.glossary_dict)} terms into dynamic glossary matcher.")

    def translate(
        self,
        text: str,
        context: Optional[List[Tuple[str, str]]] = None,
        is_final: bool = True,
    ) -> str:
        """
        Translate Japanese text to English with 2-turn rolling context.
        If is_final=False (draft), uses shorter timeout and lower token budget without updating context.
        """
        if not text or not text.strip():
            return ""

        if context is None:
            context = self.context

        recent_context = context[-2:]

        # 1. System Prompt (Static for KV Cache hit)
        messages = [{"role": "system", "content": self.system_prompt}]

        # 2. Conversation history (Clean text without old ephemeral glossary headers)
        for ja, en in recent_context:
            if ja and en:
                messages.append({"role": "user", "content": ja})
                messages.append({"role": "assistant", "content": en})

        # 3. Current User Turn with Dynamic Glossary
        matched_glossary = extract_matched_glossary(text, self.glossary_dict)
        if matched_glossary:
            glossary_snippet = "\n".join(f"- {ja}: {en}" for ja, en in matched_glossary)
            user_content = f"Glossary:\n{glossary_snippet}\n\nJapanese:\n{text}"
        else:
            user_content = text

        messages.append({"role": "user", "content": user_content})

        max_tokens = TRANSLATION_MAX_TOKENS if is_final else TRANSLATION_MAX_TOKENS_DRAFT
        timeout_sec = TRANSLATION_TIMEOUT_SEC if is_final else TRANSLATION_TIMEOUT_DRAFT_SEC

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

            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            choice = resp_data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            translated = (msg.get("content") or "").strip()

            # Clean any leftover surrounding quotes
            if (translated.startswith('"') and translated.endswith('"')) or (
                translated.startswith("'") and translated.endswith("'")
            ):
                translated = translated[1:-1].strip()

            # Update rolling context ONLY on final commits
            if translated and is_final:
                self.context.append((text, translated))
                if len(self.context) > 2:
                    self.context = self.context[-2:]

            return translated

        except urllib.error.URLError as uerr:
            if is_final:
                print(f"[translate error] Connection to local LLM failed ({uerr.reason}). Is llama-server running?", file=sys.stderr)
            return ""
        except TimeoutError:
            # Drop slow drafts silently
            return ""
        except Exception as exc:
            if is_final:
                print(f"[translate error] {type(exc).__name__}: {exc}", file=sys.stderr)
            return ""


def run_translation_worker(
    transcript_queue: queue.Queue,
    subtitle_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread: reads (epoch_id, ja_text, asr_latency, audio_dur, is_final) from transcript_queue,
    calls translate(), and puts (en_text, ja_text, total_latency, is_final) into subtitle_queue.
    Features:
    - Preemption / Drop Stale Drafts:
      If a final commit is pending, drops all drafts in transcript_queue instantly.
      If multiple drafts are queued, keeps only the latest draft.
    """
    translator = LiveTranslator()
    committed_epoch = -1

    while not stop_event.is_set():
        try:
            item = transcript_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Inspect and drain pending transcripts
        pending_items = [item]
        while not transcript_queue.empty():
            try:
                pending_items.append(transcript_queue.get_nowait())
            except queue.Empty:
                break

        # Check for any Final commit item
        final_item = None
        for it in reversed(pending_items):
            is_fin = it[4] if len(it) == 5 else it[3]
            if is_fin:
                final_item = it
                break

        if final_item is not None:
            # Prioritize Final commit immediately and drop pending drafts
            chosen_item = final_item
        else:
            chosen_item = pending_items[-1]

        if len(chosen_item) == 5:
            epoch_id, ja_text, asr_latency, audio_dur, is_final = chosen_item
        else:
            epoch_id = 0
            ja_text, asr_latency, audio_dur, is_final = chosen_item

        # Discard stale draft if its epoch was already committed
        if not is_final and epoch_id <= committed_epoch:
            continue

        t0 = time.time()
        en_text = translator.translate(ja_text, is_final=is_final)
        trans_latency = time.time() - t0

        if en_text:
            total_latency = asr_latency + trans_latency
            tag = "FINAL" if is_final else "DRAFT"
            print(f"[TRANSLATE {tag}] (epoch {epoch_id} | {trans_latency:.2f}s | total: {total_latency:.2f}s)")
            print(f"                 JA: {ja_text}")
            print(f"                 EN: {en_text}")
            subtitle_queue.put((en_text, ja_text, total_latency, is_final))

            if is_final:
                committed_epoch = max(committed_epoch, epoch_id)


if __name__ == "__main__":
    translator = LiveTranslator()
    sample_text = "だから非同期処理ってプロミスチケットみたいなのを発行してたんですよ。"
    print(f"Testing local translation with: {sample_text}")
    output = translator.translate(sample_text)
    print(f"Result: {output}")
