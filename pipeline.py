"""
Pipeline wiring.

Connects the transport layer to the voice agent.
Handles audio flow from WebRTC -> VAD -> STT -> LLM -> TTS -> WebRTC.
"""

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
        Transport receives frames -> pipeline._on_audio(frame)
        -> agent.push_audio(frame)
        -> agent processes VAD -> STT -> LLM -> TTS
        -> TTS audio -> pipeline._on_tts_audio(audio)
        -> transport.send_audio(audio)
    """

    def __init__(self, agent: VoiceAgent):
        self._agent = agent
        self._audio_output = AudioOutput()
        self._vad_model = None
        self._transport_audio_callback = None
        self._transport_send_callback = None

    def set_transport_callbacks(
        self,
        on_audio_received: callable,
        on_audio_to_send: callable,
    ) -> None:
        """
        Set callbacks from the transport layer.

        :param on_audio_received: Called with each incoming audio frame (numpy array)
        :param on_audio_to_send: Called to request audio frames to send (returns numpy array)
        """
        self._transport_audio_callback = on_audio_received
        self._transport_send_callback = on_audio_to_send

    def start(self) -> None:
        """Start the pipeline and load models."""
        # Load VAD
        self._vad_model = load_vad()
        logger.info("VAD loaded")

        # Set up agent callbacks
        self._agent.set_callbacks(
            on_listening=self._on_listening,
            on_speaking=self._on_speaking,
            on_error=self._on_error,
            on_audio_output=self._on_tts_audio,
        )

        # Start agent
        self._agent.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._agent.stop()
        self._audio_output.close()
        logger.info("Pipeline stopped")

    def on_audio_received(self, frame: np.ndarray) -> None:
        """Handle incoming audio from transport."""
        if frame is None or len(frame) == 0:
            return
        # Ensure float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        # Resample if needed (assume 16kHz)
        self._agent.push_audio(frame)

    def _on_listening(self) -> None:
        if self._transport_send_callback:
            self._transport_send_callback("listening")

    def _on_speaking(self) -> None:
        if self._transport_send_callback:
            self._transport_send_callback("speaking")

    def _on_error(self, error: str) -> None:
        logger.error(f"Agent error: {error}")
        if self._transport_send_callback:
            self._transport_send_callback(f"error:{error}")

    def _on_tts_audio(self, audio: np.ndarray) -> None:
        """Handle TTS audio from agent and queue for transport."""
        if audio is None or len(audio) == 0:
            return
        # Ensure correct sample rate (TTS is 24kHz, we may need 16kHz)
        # For now, assume transport handles resampling or we send as-is
        self._audio_output.enqueue(audio)

    def get_audio_to_send(self, max_frames: int = 160) -> np.ndarray:
        """Get audio frames to send to the transport."""
        return self._audio_output.pop_frames(max_frames)

    def has_audio_to_send(self) -> bool:
        return self._audio_output.has_audio()

    def clear_audio(self) -> None:
        self._audio_output.clear()

    def interrupt(self) -> None:
        self._agent.interrupt()
        self._audio_output.clear()
