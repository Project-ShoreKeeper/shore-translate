import time
import queue
import threading
from typing import Optional
import numpy as np

import faster_whisper.vad as vad_module
from config import (
    SAMPLE_RATE,
    VAD_SILENCE_THRESHOLD_SEC,
    VAD_MAX_SEGMENT_SEC,
    VAD_PADDING_SEC,
    VAD_THRESHOLD,
    VAD_MIN_SEGMENT_SEC,
    DRAFT_INTERVAL_SEC,
    MIN_DRAFT_AUDIO_SEC,
)

# Silero VAD v6 parameters
WINDOW_SIZE_SAMPLES = 512  # 32ms at 16kHz
CONTEXT_SIZE_SAMPLES = 64  # 4ms context
PADDING_SAMPLES = int(SAMPLE_RATE * VAD_PADDING_SEC)  # ~3200 samples (0.2s)
SILENCE_THRESHOLD_SAMPLES = int(SAMPLE_RATE * VAD_SILENCE_THRESHOLD_SEC)  # ~9600 samples (0.6s)
MAX_SEGMENT_SAMPLES = int(SAMPLE_RATE * VAD_MAX_SEGMENT_SEC)  # ~128000 samples (8.0s)


class VADSegmenter:
    """
    Streaming Silero VAD segmenter running on CPU via ONNX Runtime (C++ engine).
    Receives raw 16kHz audio chunks, groups speech into segments with pre/post padding,
    emits live intermediate drafts during speech (Option B), and closes segments on silence or max duration limit.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold_samples: int = SILENCE_THRESHOLD_SAMPLES,
        max_segment_samples: int = MAX_SEGMENT_SAMPLES,
        padding_samples: int = PADDING_SAMPLES,
        vad_threshold: float = VAD_THRESHOLD,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_samples = silence_threshold_samples
        self.max_segment_samples = max_segment_samples
        self.padding_samples = padding_samples
        self.vad_threshold = vad_threshold
        self.draft_interval_samples = int(sample_rate * DRAFT_INTERVAL_SEC)
        self.min_draft_samples = int(sample_rate * MIN_DRAFT_AUDIO_SEC)

        # Load Silero VAD ONNX model (runs on CPU via ONNX Runtime)
        self.vad_model = vad_module.get_vad_model()
        self.reset_states()

    def reset_states(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.last_context = np.zeros(CONTEXT_SIZE_SAMPLES, dtype=np.float32)
        self.is_speech = False
        self.current_segment: list[np.ndarray] = []
        self.current_segment_samples = 0
        self.consecutive_silence_samples = 0
        self.samples_since_last_draft = 0
        self.pre_buffer: list[np.ndarray] = []
        self.pre_buffer_samples = 0
        self.input_buffer = np.array([], dtype=np.float32)

    def _infer_frame(self, frame: np.ndarray) -> float:
        """Run a single 512-sample frame through the ONNX model."""
        inp = np.concatenate([self.last_context, frame]).reshape(1, -1)
        self.last_context = frame[-CONTEXT_SIZE_SAMPLES:].copy()
        out, self.h, self.c = self.vad_model.session.run(
            None, {"input": inp, "h": self.h, "c": self.c}
        )
        return float(out[0][0]) if out.ndim > 1 else float(out[0])

    def process_samples(self, samples: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
        """
        Process incoming audio samples and return any closed segments as
        (segment_audio_float32, duration_sec, is_final).
        If is_final=False, it is a progressive live draft during active speech.
        """
        if len(samples) == 0:
            return []

        self.input_buffer = np.concatenate([self.input_buffer, samples])
        closed_segments = []

        while len(self.input_buffer) >= WINDOW_SIZE_SAMPLES:
            frame = self.input_buffer[:WINDOW_SIZE_SAMPLES]
            self.input_buffer = self.input_buffer[WINDOW_SIZE_SAMPLES:]
            prob = self._infer_frame(frame)

            if prob >= self.vad_threshold:
                # Speech detected
                if not self.is_speech:
                    # Speech started: prepend pre-speech padding buffer
                    self.is_speech = True
                    self.current_segment = list(self.pre_buffer)
                    self.current_segment_samples = self.pre_buffer_samples
                    self.consecutive_silence_samples = 0
                    self.samples_since_last_draft = 0
                    self.pre_buffer.clear()
                    self.pre_buffer_samples = 0

                self.current_segment.append(frame)
                self.current_segment_samples += len(frame)
                self.samples_since_last_draft += len(frame)
                self.consecutive_silence_samples = 0

                # Check max length constraint (final commit)
                if self.current_segment_samples >= self.max_segment_samples:
                    seg_audio = np.concatenate(self.current_segment).astype(np.float32)
                    closed_segments.append((seg_audio, len(seg_audio) / self.sample_rate, True))
                    self.reset_states()
                # Check draft interval (live intermediate draft)
                elif (
                    self.current_segment_samples >= self.min_draft_samples
                    and self.samples_since_last_draft >= self.draft_interval_samples
                ):
                    draft_audio = np.concatenate(self.current_segment).astype(np.float32)
                    closed_segments.append((draft_audio, len(draft_audio) / self.sample_rate, False))
                    self.samples_since_last_draft = 0

            else:
                # Silence detected
                if self.is_speech:
                    self.current_segment.append(frame)
                    self.current_segment_samples += len(frame)
                    self.samples_since_last_draft += len(frame)
                    self.consecutive_silence_samples += len(frame)

                    # Check silence threshold (final commit)
                    if self.consecutive_silence_samples >= self.silence_threshold_samples:
                        excess_silence = self.consecutive_silence_samples - self.padding_samples
                        seg_audio = np.concatenate(self.current_segment).astype(np.float32)
                        if excess_silence > 0 and len(seg_audio) > excess_silence:
                            seg_audio = seg_audio[:-excess_silence]

                        duration = len(seg_audio) / self.sample_rate
                        # Keep only segments of meaningful duration
                        if duration >= VAD_MIN_SEGMENT_SEC:
                            closed_segments.append((seg_audio, duration, True))

                        self.reset_states()
                    elif self.current_segment_samples >= self.max_segment_samples:
                        seg_audio = np.concatenate(self.current_segment).astype(np.float32)
                        closed_segments.append((seg_audio, len(seg_audio) / self.sample_rate, True))
                        self.reset_states()
                else:
                    # Maintain rolling pre-speech buffer
                    self.pre_buffer.append(frame)
                    self.pre_buffer_samples += len(frame)
                    while self.pre_buffer_samples > self.padding_samples and len(self.pre_buffer) > 1:
                        removed = self.pre_buffer.pop(0)
                        self.pre_buffer_samples -= len(removed)

        return closed_segments

    def flush(self) -> list[tuple[np.ndarray, float, bool]]:
        """Flush any currently ongoing speech segment."""
        closed_segments = []
        if self.is_speech and len(self.current_segment) > 0:
            seg_audio = np.concatenate(self.current_segment).astype(np.float32)
            duration = len(seg_audio) / self.sample_rate
            if duration >= VAD_MIN_SEGMENT_SEC:
                closed_segments.append((seg_audio, duration, True))
        self.reset_states()
        return closed_segments


def run_vad_worker(
    audio_queue: queue.Queue,
    speech_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread: reads raw audio from audio_queue, applies VAD,
    and pushes speech segments (np.ndarray float32, duration_sec, is_final) into speech_queue.
    """
    segmenter = VADSegmenter()

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        segments = segmenter.process_samples(chunk)
        for seg_audio, duration, is_final in segments:
            speech_queue.put((seg_audio, duration, is_final))
