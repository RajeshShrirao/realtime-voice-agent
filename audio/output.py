"""
Audio output layer.

In V1, synthesized TTS audio is sent to the WebRTC transport as PCM frames.
This module manages the output queue and provides frames to the transport.
"""

import collections
import numpy as np
import threading

from audio.formats import TTS_SAMPLE_RATE, GAP_SAMPLES, FADE_SAMPLES


class AudioOutput:
    """Queues TTS audio chunks and provides them as PCM frames for WebRTC."""

    def __init__(self):
        self._queue: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._closed = False

    def enqueue(self, audio: np.ndarray) -> None:
        """Add a TTS audio chunk to the output queue."""
        if self._closed:
            return
        if len(audio) == 0:
            return
        # Apply fade in/out
        n = min(FADE_SAMPLES, len(audio) // 4)
        if n >= 2:
            audio = audio.copy()
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            audio[:n] *= ramp
            audio[-n:] *= ramp[::-1]
        with self._lock:
            self._queue.append(audio.copy())

    def pop_frames(self, max_frames: int) -> np.ndarray:
        """Pop up to max_frames samples from the queue."""
        with self._lock:
            if not self._queue:
                return np.array([], dtype=np.float32)
            first = self._queue[0]
            n = min(len(first), max_frames)
            result = first[:n].copy()
            if n < len(first):
                self._queue[0] = first[n:]
            else:
                self._queue.popleft()
            return result

    def has_audio(self) -> bool:
        with self._lock:
            return len(self._queue) > 0

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def close(self) -> None:
        self._closed = True
        self.clear()
