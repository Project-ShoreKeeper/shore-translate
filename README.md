# shore-translate

Real-time desktop application for translating Japanese livestreams into English subtitles. It runs **100% locally on your machine** with zero cloud API keys, zero PyTorch overhead, and native C++ GPU acceleration.

Captures system audio via loopback, detects speech boundaries with Silero VAD, transcribes Japanese speech with faster-whisper on CUDA, translates it to natural English using a local LLM (llama.cpp / llama-server), and displays crisp subtitles in a floating overlay window.

---

## 100% Local Architecture (Pure C++ Engine Stack)

The entire AI pipeline runs locally across separate threads connected by thread-safe `queue.Queue` buffers:

```
System Audio (soundcard loopback)
   ↓ 16 kHz mono float32 chunks (~30ms)
VAD Adaptive Soft-Cut & Draft Emission (CPU - ONNX Runtime C++ engine)
   ├─ Every ~0.85s: Live speech draft audio (Streaming Draft)
   ├─ Soft-Cut [4.0s - 6.0s]: 0.15s micro-pause detection (catches breath / clause boundary)
   └─ On 0.40s silence / 6.0s hard limit: Final commit + Epoch increment
ASR Transcription (CUDA - faster-whisper / CTranslate2 C++ engine)
   ├─ Drop Stale Drafts: Preempts and purges obsolete drafts when Final arrives
   └─ Japanese text + hallucination filters + glossary injection
Translation (Local llama-server / llama.cpp on CUDA)
   ├─ 100% Static System Prompt: 100% Prefix KV Cache hit (--cache-reuse in llama-server)
   ├─ Dynamic Glossary Matching: Ultra-fast (<0.05ms) keyword extraction in User turn
   └─ English subtitle text (60 tokens draft / 120 tokens final)
Floating Overlay (PySide6 GUI)
   ├─ Amber Italic: Live draft preview
   └─ Crisp White: Final committed translation
```

### Key Highlights
- **Zero Cloud API Keys**: Translation runs on your local `llama-server` (baseline: **Qwen2.5-7B-Instruct** on port `9090`).
- **Streaming Draft & Revise**: Live partial translations update every ~0.85s (in amber italic) as the streamer speaks, then instantly lock in with high-precision translation (in white bold) when a sentence completes.
- **Drop Stale Drafts & Preemption**: When a Final Commit occurs, all pending intermediate drafts are purged instantly from queues and dropped by workers, ensuring zero queue lag for final subtitles.
- **Adaptive Soft-Cut VAD**: Rather than cutting speech rigidly at 5.0s, speech between 4.0s and 6.0s dynamically uses an ultra-sensitive 0.15s micro-pause threshold to cleanly cut between clauses (e.g. 〜て、〜けど) without slicing Japanese words in half.
- **100% Prefix KV Cache Reuse + Dynamic Glossary**: The System Prompt is completely invariant, enabling llama.cpp to reuse 100% of prefix KV cache (Time-To-First-Token < 20ms). Only terms actually present in the current sentence are dynamically injected into the user turn.
- **Zero PyTorch Dependency**: Both VAD (ONNX Runtime) and ASR (CTranslate2) run on native C++ engines, avoiding GIL locks and PyTorch memory overhead.
- **Hardware Optimized**: Fits comfortably inside 16 GB VRAM on NVIDIA RTX GPUs (Whisper ~3.9 GB + LLM ~4-10 GB = ~8-14 GB total).
- **Audio Capture**: `soundcard` captures loopback monitor audio directly from the default speaker (PipeWire / PulseAudio).
- **Linux Wayland/X11 Compatibility**: Auto-detects Xwayland authentication (`xcb` / `_NET_WM_STATE_ABOVE`) so the overlay remains reliably floating on top of browsers and media players.
- **Subtitle Overlay**: `PySide6` frameless, always-on-top, draggable, white text with dark outline, 2-line limit, auto-clearing after 6 seconds of silence.

---

## Requirements

- **OS**: Linux (tested on Ubuntu 26.04 LTS / PipeWire / Wayland / X11)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support (e.g. RTX 5060 Ti / RTX 4070+ with 12-16 GB VRAM)
- **Local Model Server**: `llama-server` running on `http://localhost:9090` (baseline: Qwen2.5-7B)

---

## Quick Start

### 1. Start the Local Translation Server
In a separate terminal or as a service, start your local `llama-server` with the baseline model:
```bash
/mnt/external_ssd/start_llama_server.sh qwen-7b
```
*(Baseline: `qwen-7b` (Qwen2.5-7B-Instruct Q4_K_M) is the primary baseline model, delivering accurate technical translation, natural conversational fluency, and strict glossary adherence. Alternatives: `qwen-3b` / `cat-1.4b` for ultra-low latency (~100ms) or `gemma-12b` for nuanced expressions).*

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Translator App
```bash
python main.py
```
- Open any Japanese programming / tech livestream (e.g. YouTube, Twitch) in your browser.
- Subtitles will appear automatically at the bottom of the screen.
- **Move**: Click and drag the subtitle window anywhere on your screen.
- **Close**: Press `Esc` or click the small `✕` button in the top-right corner.

---

## Offline Benchmark & Testing

You can benchmark latency across all stages (VAD, ASR, LLM) without playing live audio:
```bash
python benchmark_latency.py
```
This processes `sample_japanese_speech.wav` and reports exact millisecond breakdown for VAD processing, Whisper transcription, and local LLM translation.

---

## Project Structure

```
shore-translate/
  main.py                    # Starts worker threads and PySide6 overlay window
  audio.py                   # System audio loopback capture (soundcard)
  vad.py                     # Silero VAD speech segmentation & draft trigger (ONNX Runtime)
  asr.py                     # faster-whisper transcription on CUDA (CTranslate2)
  translate.py               # Local LLM translation with glossary & 2-turn context
  overlay.py                 # Frameless, draggable, always-on-top subtitle window
  config.py                  # Central configuration constants
  glossary.txt               # Editable technical term pairs (日本語 = English)
  benchmark_latency.py       # Latency benchmark script for VAD, Whisper, and LLM
  sample_japanese_speech.wav # 16kHz test speech audio for offline verification
  requirements.txt           # Python dependencies (no torch, no external cloud SDKs)
  .gitignore                 # Git ignore rules (.venv, cache, temporary recordings)
  README.md                  # Documentation and usage guide
```

---

## Configuration

Edit `config.py` to customize parameters:
- `LOCAL_LLM_URL`: Endpoint for translation (default `"http://localhost:9090/v1/chat/completions"`).
- `ASR_MODEL_NAME`: Default `"large-v3"`. Options: `"kotoba-tech/kotoba-whisper-v2.0-faster"`, `"deepdml/faster-whisper-large-v3-turbo-ct2"`.
- `VAD_SILENCE_THRESHOLD_SEC`: Standard silence duration to close a segment (default `0.40` s).
- `VAD_SOFT_MAX_SEC`: Speech duration threshold where adaptive silence kicks in (default `4.0` s).
- `VAD_HARD_MAX_SEC`: Hard maximum segment duration before forced cut (default `6.0` s).
- `VAD_MICRO_PAUSE_SEC`: Micro-pause threshold for natural clause/breath soft-cut in [4.0s, 6.0s] (default `0.15` s).
- `DRAFT_INTERVAL_SEC`: Emit live intermediate draft every `0.85` s during speech.
- `MIN_DRAFT_AUDIO_SEC`: Minimum speech duration (`0.65` s) before emitting the first draft.
- `SUBTITLE_CLEAR_DELAY_SEC`: Delay before fading clear subtitles (default `6.0` s).
- `OVERLAY_MAX_LINES`: Subtitle display limit (default `2` lines).

---

## Glossary (`glossary.txt`)

Add custom IT/programming terms in `日本語 = English` format:
```text
リポジトリ = repository
プルリクエスト = pull request
デプロイ = deploy
コミット = commit
ブランチ = branch
マージ = merge
...
```
- **ASR Acoustic Biasing**: Japanese terms are loaded at startup into Whisper `initial_prompt` and `hotwords`.
- **Dynamic Matching & KV Cache Protection**: During translation, an ultra-fast (<0.05ms) matcher scans each Japanese sentence for glossary keywords. Matched definitions are injected directly into that specific User turn rather than mutating the System Prompt, preserving 100% prefix KV cache reuse while allowing `glossary.txt` to scale to thousands of terms.
