from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
GLOSSARY_PATH = BASE_DIR / "glossary.txt"

# Audio capture
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "float32"
CHUNK_DURATION_SEC = 0.03  # ~30ms chunk size for VAD processing (~480 samples at 16kHz)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)

# VAD & Streaming settings (Silero VAD) - Option B: Streaming Draft & Revise
VAD_SILENCE_THRESHOLD_SEC = 0.40  # Final segment close after ~0.40s silence
VAD_MAX_SEGMENT_SEC = 5.0         # Maximum segment length ~5.0s
VAD_PADDING_SEC = 0.20            # Keep ~0.20s padding
VAD_MIN_SEGMENT_SEC = 0.40        # Minimum speech duration for final commit
VAD_THRESHOLD = 0.5               # Speech probability threshold

# Option B Streaming Draft parameters
DRAFT_INTERVAL_SEC = 0.85         # Emit live translation draft every ~0.85s during active speech
MIN_DRAFT_AUDIO_SEC = 0.65        # Require at least 0.65s audio before emitting the first draft

# ASR settings (faster-whisper)
ASR_MODEL_NAME = "large-v3"       # Alternative: "kotoba-tech/kotoba-whisper-v2.0-faster"
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

# Translation settings (Local llama.cpp / llama-server or OpenAI-compatible endpoint)
LOCAL_LLM_URL = "http://localhost:9090/v1/chat/completions"  # Local llama-server endpoint
TRANSLATION_MODEL = "default"                                # Model loaded in llama-server
TRANSLATION_TIMEOUT_SEC = 5.0
TRANSLATION_MAX_TOKENS = 120

# Subtitle overlay settings
SUBTITLE_CLEAR_DELAY_SEC = 6.0
OVERLAY_MAX_LINES = 2
