"""
Audio input layer.

In V1, audio comes from the WebRTC transport as PCM frames.
This module will buffer frames, run VAD, and produce transcribed text
via the STT service.

For now, this is a placeholder that defines the interface.
"""

import numpy as np
from typing import Callable

from services.vad import is_speech
from audio.formats import SAMPLE_RATE, VAD_CHUNK_SAMPLES, PRE_ROLL_CHUNKS


class AudioInput:
    """Bufffers incoming audio frames and detects speech via VAD."""

    def __init__(self, vad_model, on_speech_detected: Callable[[np.ndarray], None]):
        self._vad_model = vad_model
        self._on_speech_detected = on_speech_detected
        self._buffer: list[np.ndarray] = []
        self._speech_samples: list[np.ndarray] = []
        self._speaking = False
        self._silence_streak = 0

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a single audio frame (VAD_CHUNK_SAMPLES length)."""
        # VAD check
        speaking = is_speech(frame, self._vad_model, SAMPLE_RATE)

        if speaking:
            self._speaking = True
            self._silence_streak = 0
            self._speech_samples.append(frame.copy())
        elif self._speaking:
            self._silence_streak += 1

        # Keep buffer bounded
        self._buffer.append(frame.copy())
        if len(self._buffer) > 100:
            self._buffer = self._buffer[-50:]

    def get_buffered_audio(self) -> np.ndarray:
        """Return all buffered audio as a single numpy array."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer)

    def is_speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._buffer.clear()
        self._speech_samples.clear()
        self._speaking = False
        self._silence_streak = 0
