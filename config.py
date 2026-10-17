import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
GLOSSARY_PATH = BASE_DIR / "glossary.txt"
DEBUG_ASR_DIR = BASE_DIR / "debug_asr"

# Audio capture
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "float32"
CHUNK_DURATION_SEC = 0.03  # ~30ms chunk size for VAD processing (~480 samples at 16kHz)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)

# VAD & Streaming settings (Silero VAD) - Adaptive Soft-Cut & Draft
VAD_SILENCE_THRESHOLD_SEC = 0.40  # Standard silence duration to close segment (< 4.0s)
VAD_SOFT_MAX_SEC = 4.0            # Speech duration threshold where adaptive silence kicks in
VAD_HARD_MAX_SEC = 6.0            # Hard ceiling max segment duration before forced cut
VAD_MICRO_PAUSE_SEC = 0.15        # Micro-pause threshold (breath/clause boundary) for soft-cut in [4.0s, 6.0s]
VAD_PADDING_SEC = 0.20            # Keep ~0.20s padding before speech
VAD_MIN_SEGMENT_SEC = 0.40        # Minimum speech duration for final commit
VAD_THRESHOLD = 0.5               # Speech probability threshold

# Option B Streaming Draft parameters
DRAFT_INTERVAL_SEC = 0.85         # Emit live translation draft every ~0.85s during active speech
MIN_DRAFT_AUDIO_SEC = 0.65        # Require at least 0.65s audio before emitting the first draft

# ASR settings (faster-whisper)
# Options:
# - "kotoba-tech/kotoba-whisper-v2.0-faster": Japanese distil model, ultra fast (4 decoder layers, default)
# - "deepdml/faster-whisper-large-v3-turbo-ct2": Whisper Turbo, 3-4x faster than large-v3
# - "large-v3": Maximum accuracy (32 decoder layers)
ASR_MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "kotoba-tech/kotoba-whisper-v2.0-faster")
ASR_DEVICE = "cuda"
ASR_COMPUTE_TYPE = "float16"
ASR_LANGUAGE = "ja"
ASR_BEAM_SIZE = 1
ASR_TEMPERATURE = 0.0
ASR_CONDITION_ON_PREVIOUS_TEXT = False
ASR_VAD_FILTER = True

# ASR Hallucination filtering
ASR_MIN_CHAR_LENGTH = 2
ASR_MAX_NO_SPEECH_PROB = 0.6
ASR_MIN_AVG_LOGPROB = -1.0

# ASR Verification & Telemetry Suite
ENABLE_ASR_TELEMETRY = os.environ.get("ENABLE_ASR_TELEMETRY", "true").lower() in ("true", "1", "yes")
ENABLE_LLM_ASR_CRITIC = os.environ.get("ENABLE_LLM_ASR_CRITIC", "false").lower() in ("true", "1", "yes")
DEBUG_SAVE_LOW_CONF_AUDIO = os.environ.get("DEBUG_SAVE_LOW_CONF_AUDIO", "false").lower() in ("true", "1", "yes")
LOW_CONF_THRESHOLD = -0.65
HIGH_CONF_THRESHOLD = -0.35

# Translation settings (Local llama.cpp / llama-server or OpenAI-compatible endpoint)
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:9090/v1/chat/completions")
TRANSLATION_MODEL = "default"                                # Model loaded in llama-server
TRANSLATION_TIMEOUT_SEC = 5.0                                # Timeout for final commit translation
TRANSLATION_TIMEOUT_DRAFT_SEC = 1.5                          # Short timeout for draft to prevent queue blockage
TRANSLATION_MAX_TOKENS = 120
TRANSLATION_MAX_TOKENS_DRAFT = 60

# Subtitle overlay settings
SUBTITLE_CLEAR_DELAY_SEC = 6.0
OVERLAY_MAX_LINES = 2
