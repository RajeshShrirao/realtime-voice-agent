"""Pipeline wiring.

Connects the transport layer to the voice agent.
Handles audio flow from WebRTC -> VAD -> STT -> LLM -> TTS -> WebRTC.
"""

import asyncio
import logging
import numpy as np
import threading

from agent import VoiceAgent
from audio.formats import SAMPLE_RATE, TTS_SAMPLE_RATE
from audio.output import AudioOutput
from services.vad import load_vad

logger = logging.getLogger("pipeline")


class Pipeline:
    """
    Wires the transport to the voice agent.

    Audio flow:
        Transport receives frames -> pipeline.on_audio_received(frame)
        -> agent.push_audio(frame)
        -> agent processes VAD -> STT -> LLM -> TTS
        -> TTS audio -> pipeline._on_tts_audio(audio)
        -> audio_queue.put(audio) -> AudioSenderTrack -> browser
    """

    def __init__(self, agent: VoiceAgent):
        self._agent = agent
        self._audio_output = AudioOutput()
        self._vad_model = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._lock = threading.Lock()

    def set_audio_sender_queue(self, queue: asyncio.Queue) -> None:
        """Set the asyncio queue for sending TTS audio to the browser via WebRTC."""
        with self._lock:
            self._audio_queue = queue
        logger.info("Pipeline: audio sender queue set")

    def start(self) -> None:
        """Start the pipeline and load models."""
        self._vad_model = load_vad()
        logger.info("VAD loaded")

        self._agent.set_callbacks(
            on_listening=self._on_listening,
            on_speaking=self._on_speaking,
            on_error=self._on_error,
            on_audio_output=self._on_tts_audio,
        )

        self._agent.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._agent.stop()
        self._audio_output.clear()
        logger.info("Pipeline stopped")

    def on_audio_received(self, frame: np.ndarray) -> None:
        """Handle incoming audio from WebRTC transport."""
        if frame is None or len(frame) == 0:
            return
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        self._agent.push_audio(frame)

    def _on_listening(self) -> None:
        logger.info("Status: listening")

    def _on_speaking(self) -> None:
        logger.info("Status: speaking")

    def _on_error(self, error: str) -> None:
        logger.error(f"Agent error: {error}")

    def _on_tts_audio(self, audio: np.ndarray) -> None:
        """Handle TTS audio from agent and queue for WebRTC send."""
        if audio is None or len(audio) == 0:
            return
        # Send to WebRTC sender queue if available
        with self._lock:
            if self._audio_queue is not None:
                try:
                    self._audio_queue.put_nowait(audio)
                except asyncio.QueueFull:
                    logger.warning("Audio queue full, dropping frame")
                except RuntimeError:
                    # Queue might be closed
                    pass
        # Also keep local output for potential future use
        self._audio_output.enqueue(audio)

    def interrupt(self) -> None:
        """Interrupt current TTS and clear audio queue."""
        self._agent.interrupt()
        self._audio_output.clear()
        logger.info("Pipeline: interrupted")

    @property
    def has_audio_output(self) -> bool:
        return self._audio_output.has_audio()
