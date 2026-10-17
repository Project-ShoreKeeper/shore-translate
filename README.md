# shore-translate

Real-time desktop application for translating Japanese livestreams into English subtitles. It runs **100% locally on your machine** with zero cloud API keys, zero PyTorch overhead, and native C++ GPU acceleration.

Captures system audio via loopback, detects speech boundaries with Silero VAD, transcribes Japanese speech with faster-whisper on CUDA, translates it to natural English using a local LLM (llama.cpp / llama-server), and displays crisp subtitles in a floating overlay window.

---

## 100% Local Architecture (Pure C++ Engine Stack)

The entire AI pipeline runs locally across separate threads connected by thread-safe `queue.Queue` buffers:

```
System Audio (soundcard loopback)
   ↓ 16 kHz mono float32 chunks
VAD Segmentation (CPU - ONNX Runtime C++ engine)
   ↓ speech segments (0.2s padding, 0.6s silence / 8s max)
ASR Transcription (CUDA - faster-whisper / CTranslate2 C++ engine)
   ↓ Japanese text + latency stats
Translation (Local llama-server / llama.cpp on CUDA)
   ↓ English subtitle text
Floating Overlay (PySide6 GUI)
```

### Key Highlights
- **Zero Cloud API Keys**: Translation runs on your local `llama-server` (e.g. Gemma4-12B on port `9090`).
- **Zero PyTorch Dependency**: Both VAD (ONNX Runtime) and ASR (CTranslate2) run on native C++ engines.
- **Hardware Optimized**: Fits comfortably inside 16 GB VRAM on NVIDIA RTX GPUs (Whisper ~3.9 GB + Gemma-12B ~9.6 GB = ~13.5 GB total).
- **Audio Capture**: `soundcard` captures loopback monitor audio directly from the default speaker (PipeWire / PulseAudio).
- **Subtitle Overlay**: `PySide6` frameless, always-on-top, draggable, white text with dark outline, 2-line limit, auto-clearing after 6 seconds of silence.

---

## Requirements

- **OS**: Linux (tested on Ubuntu 26.04 LTS / PipeWire / Wayland / X11)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support (e.g. RTX 5060 Ti with 16 GB VRAM)
- **Local Model Server**: `llama-server` running on `http://localhost:9090` (started via `/mnt/external_ssd/start_llama_server.sh`)

---

## Quick Start

### 1. Start the Local Translation Server
In a separate terminal or as a service, start your local `llama-server`:
```bash
/mnt/external_ssd/start_llama_server.sh [qwen-3b|cat-1.4b|cat-3.3b|qwen-7b|gemma-12b]
```
*(Default: `qwen-3b` for ultra-fast ~100ms translation responses, low VRAM footprint, and strict technical glossary adherence. Use `cat-1.4b` for maximum throughput or `cat-3.3b` / `qwen-7b` for higher translation nuance).*

### 2. Start the Translator App
```bash
python main.py
```
- Open any Japanese programming / tech livestream (e.g. YouTube, Twitch) in your browser.
- Subtitles will appear automatically at the bottom of the screen.
- **Move**: Click and drag the subtitle window anywhere on your screen.
- **Close**: Press `Esc` or click the small `✕` button in the top-right corner.

---

## Project Structure

```
shore-translate/
  main.py          # Starts worker threads and PySide6 overlay window
  audio.py         # System audio loopback capture (soundcard)
  vad.py           # Silero VAD speech segmentation (CPU via ONNX Runtime)
  asr.py           # faster-whisper transcription on CUDA (CTranslate2)
  translate.py     # Local LLM translation with glossary & 2-turn context
  overlay.py       # Frameless, draggable, always-on-top subtitle window
  config.py        # Central configuration constants
  glossary.txt     # Editable technical term pairs (日本語 = English)
  requirements.txt # Dependencies (no torch, no external cloud SDKs)
  README.md        # Documentation and usage guide
```

---

## Configuration

Edit `config.py` to customize parameters:
- `LOCAL_LLM_URL`: Endpoint for translation (default `"http://localhost:9090/v1/chat/completions"`).
- `ASR_MODEL_NAME`: Default `"large-v3"`. Alternative: `"kotoba-tech/kotoba-whisper-v2.0-faster"`.
- `VAD_SILENCE_THRESHOLD_SEC`: Silence duration to close a segment (default `0.6` s).
- `VAD_MAX_SEGMENT_SEC`: Maximum segment duration (default `8.0` s).
- `SUBTITLE_CLEAR_DELAY_SEC`: Delay before fading clear subtitles (default `6.0` s).

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
The Japanese terms are automatically loaded at startup into ASR `initial_prompt` and `hotwords`. The full glossary is included in the translation system prompt for exact term matching.
