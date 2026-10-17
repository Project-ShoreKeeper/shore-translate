import time
import queue
import threading
from typing import Optional
import numpy as np

import faster_whisper.vad as vad_module
from config import (
    SAMPLE_RATE,
    VAD_SILENCE_THRESHOLD_SEC,
    VAD_SOFT_MAX_SEC,
    VAD_HARD_MAX_SEC,
    VAD_MICRO_PAUSE_SEC,
    VAD_PADDING_SEC,
    VAD_THRESHOLD,
    VAD_MIN_SEGMENT_SEC,
    DRAFT_INTERVAL_SEC,
    MIN_DRAFT_AUDIO_SEC,
)

# Silero VAD v6 parameters
WINDOW_SIZE_SAMPLES = 512  # 32ms at 16kHz
CONTEXT_SIZE_SAMPLES = 64  # 4ms context
PADDING_SAMPLES = int(SAMPLE_RATE * VAD_PADDING_SEC)


class VADSegmenter:
    """
    Streaming Silero VAD segmenter running on CPU via ONNX Runtime (C++ engine).
    Features:
    - Pre/post speech padding buffers
    - Progressive live draft emission every ~0.85s (Option B Streaming Draft)
    - Adaptive Soft-Cut between 4.0s and 6.0s: lowers silence threshold to 0.15s
      to catch natural breathing / clause micro-pauses (〜て、〜けど、)
    - Hard ceiling cut at 6.0s
    - Epoch ID tracking for instantaneous stale-draft preemption
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold_sec: float = VAD_SILENCE_THRESHOLD_SEC,
        soft_max_sec: float = VAD_SOFT_MAX_SEC,
        hard_max_sec: float = VAD_HARD_MAX_SEC,
        micro_pause_sec: float = VAD_MICRO_PAUSE_SEC,
        padding_samples: int = PADDING_SAMPLES,
        vad_threshold: float = VAD_THRESHOLD,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_samples = int(sample_rate * silence_threshold_sec)
        self.soft_max_samples = int(sample_rate * soft_max_sec)
        self.hard_max_samples = int(sample_rate * hard_max_sec)
        self.micro_pause_samples = int(sample_rate * micro_pause_sec)
        self.padding_samples = padding_samples
        self.vad_threshold = vad_threshold
        self.draft_interval_samples = int(sample_rate * DRAFT_INTERVAL_SEC)
        self.min_draft_samples = int(sample_rate * MIN_DRAFT_AUDIO_SEC)

        # Monotonically increasing epoch sequence counter
        self.epoch_id = 0

        # Load Silero VAD ONNX model (CPU ONNX Runtime)
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

    def process_samples(self, samples: np.ndarray) -> list[tuple[int, np.ndarray, float, bool]]:
        """
        Process incoming audio samples and return any closed segments as
        (epoch_id, segment_audio_float32, duration_sec, is_final).
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

                # 1. Hard ceiling constraint (forced final cut at 6.0s)
                if self.current_segment_samples >= self.hard_max_samples:
                    seg_audio = np.concatenate(self.current_segment).astype(np.float32)
                    closed_segments.append((self.epoch_id, seg_audio, len(seg_audio) / self.sample_rate, True))
                    self.epoch_id += 1
                    self.reset_states()

                # 2. Live intermediate draft interval check
                elif (
                    self.current_segment_samples >= self.min_draft_samples
                    and self.samples_since_last_draft >= self.draft_interval_samples
                ):
                    draft_audio = np.concatenate(self.current_segment).astype(np.float32)
                    closed_segments.append((self.epoch_id, draft_audio, len(draft_audio) / self.sample_rate, False))
                    self.samples_since_last_draft = 0

            else:
                # Silence detected
                if self.is_speech:
                    self.current_segment.append(frame)
                    self.current_segment_samples += len(frame)
                    self.samples_since_last_draft += len(frame)
                    self.consecutive_silence_samples += len(frame)

                    # Adaptive Soft-Cut threshold calculation:
                    # If speech duration reached soft_max (4.0s), require only micro_pause (0.15s)
                    # of silence to catch natural breathing / clause boundary without cutting words.
                    if self.current_segment_samples >= self.soft_max_samples:
                        required_silence = self.micro_pause_samples
                    else:
                        required_silence = self.silence_threshold_samples

                    # 1. Check silence threshold (Soft-Cut or standard silence)
                    if self.consecutive_silence_samples >= required_silence:
                        excess_silence = self.consecutive_silence_samples - self.padding_samples
                        seg_audio = np.concatenate(self.current_segment).astype(np.float32)
                        if excess_silence > 0 and len(seg_audio) > excess_silence:
                            seg_audio = seg_audio[:-excess_silence]

                        duration = len(seg_audio) / self.sample_rate
                        if duration >= VAD_MIN_SEGMENT_SEC:
                            closed_segments.append((self.epoch_id, seg_audio, duration, True))
                            self.epoch_id += 1

                        self.reset_states()

                    # 2. Check hard max ceiling
                    elif self.current_segment_samples >= self.hard_max_samples:
                        seg_audio = np.concatenate(self.current_segment).astype(np.float32)
                        closed_segments.append((self.epoch_id, seg_audio, len(seg_audio) / self.sample_rate, True))
                        self.epoch_id += 1
                        self.reset_states()
                else:
                    # Maintain rolling pre-speech buffer
                    self.pre_buffer.append(frame)
                    self.pre_buffer_samples += len(frame)
                    while self.pre_buffer_samples > self.padding_samples and len(self.pre_buffer) > 1:
                        removed = self.pre_buffer.pop(0)
                        self.pre_buffer_samples -= len(removed)

        return closed_segments

    def flush(self) -> list[tuple[int, np.ndarray, float, bool]]:
        """Flush any currently ongoing speech segment."""
        closed_segments = []
        if self.is_speech and len(self.current_segment) > 0:
            seg_audio = np.concatenate(self.current_segment).astype(np.float32)
            duration = len(seg_audio) / self.sample_rate
            if duration >= VAD_MIN_SEGMENT_SEC:
                closed_segments.append((self.epoch_id, seg_audio, duration, True))
                self.epoch_id += 1
        self.reset_states()
        return closed_segments


def run_vad_worker(
    audio_queue: queue.Queue,
    speech_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread: reads raw audio from audio_queue, applies VAD,
    and pushes speech segments (epoch_id, np.ndarray float32, duration_sec, is_final) into speech_queue.
    """
    segmenter = VADSegmenter()

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        segments = segmenter.process_samples(chunk)
        for epoch_id, seg_audio, duration, is_final in segments:
            speech_queue.put((epoch_id, seg_audio, duration, is_final))
