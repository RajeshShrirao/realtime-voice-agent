"""VoiceAgent abstraction.

The agent orchestrates the voice pipeline:
    Audio input -> VAD -> STT -> LLM -> TTS -> Audio output

The agent does not know about the transport. It only knows about
audio input/output callbacks.
"""

import logging
import numpy as np
import threading
import time

from services.vad import load_vad, is_speech
from services.stt import transcribe
from services.llm import generate_stream
from services.tts import synthesize
from audio.formats import (
    SAMPLE_RATE,
    VAD_CHUNK_SAMPLES,
    PRE_ROLL_CHUNKS,
    BARGE_IN_FRAMES,
    BARGE_IN_THRESHOLD_S,
)

logger = logging.getLogger("agent")


class VoiceAgent:
    """Generic realtime voice agent.

    Usage:
        agent = VoiceAgent(system_prompt="...")
        agent.set_cerebras_client(client)
        agent.start()
        # Push audio frames via agent.push_audio(frame)
        # Agent handles VAD -> STT -> LLM -> TTS -> output
        agent.stop()
    """

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt
        self._vad_model = None
        self._tts_model = None
        self._cerebras_client = None

        self._audio_buffer: list[np.ndarray] = []
        self._speech_samples: list[np.ndarray] = []
        self._speaking = False
        self._silence_streak = 0
        self._speech_chunk_count = 0

        self._messages: list[dict] = []
        self._tts_queue: list[str] = []
        self._tts_processing = False

        self._is_playing = False
        self._interrupted = False

        self._running = False
        self._thread: threading.Thread = threading.Thread()
        self._lock = threading.Lock()

        # Callbacks
        self._on_listening = None
        self._on_speaking = None
        self._on_error = None
        self._on_audio_output = None

        # Stats
        self._turn_count = 0

    def set_cerebras_client(self, client) -> None:
        """Set the Cerebras SDK client for LLM calls."""
        self._cerebras_client = client

    def set_callbacks(
        self,
        on_listening=None,
        on_speaking=None,
        on_error=None,
        on_audio_output=None,
    ) -> None:
        """Set callbacks for agent state changes."""
        with self._lock:
            self._on_listening = on_listening
            self._on_speaking = on_speaking
            self._on_error = on_error
            self._on_audio_output = on_audio_output

    def start(self) -> None:
        """Load models and start the agent loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("VoiceAgent started")

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def push_audio(self, frame: np.ndarray) -> None:
        """Push an audio frame from the transport."""
        if not self._running:
            return
        with self._lock:
            self._audio_buffer.append(frame.copy())
            # Keep buffer bounded
            if len(self._audio_buffer) > 200:
                self._audio_buffer = self._audio_buffer[-100:]

    def _run_loop(self) -> None:
        """Main agent loop: VAD -> STT -> LLM -> TTS."""
        while self._running:
            self._process_audio()
            time.sleep(0.01)

    def _process_audio(self) -> None:
        """Process buffered audio: VAD detection, STT, LLM, TTS."""
        with self._lock:
            if not self._audio_buffer:
                return

            recent = self._audio_buffer[-1] if self._audio_buffer else np.array([])
            if len(recent) < VAD_CHUNK_SAMPLES:
                return

            speaking = (
                is_speech(recent, self._vad_model, SAMPLE_RATE)
                if self._vad_model
                else True
            )

            if speaking:
                self._speaking = True
                self._silence_streak = 0
                self._speech_chunk_count += 1
                self._speech_samples.append(recent.copy())

                if self._on_listening:
                    self._on_listening()
            elif self._speaking:
                self._silence_streak += 1

            # Check for end of speech
            min_speech_chunks = int(SAMPLE_RATE * 0.4 / VAD_CHUNK_SAMPLES)
            max_silence_chunks = int(SAMPLE_RATE * 2.4 / VAD_CHUNK_SAMPLES)

            if (
                self._speaking
                and self._speech_chunk_count >= min_speech_chunks
                and self._silence_streak >= max_silence_chunks
            ):
                self._end_turn()
                self._speaking = False
                self._silence_streak = 0
                self._speech_chunk_count = 0
                self._speech_samples.clear()
                self._audio_buffer.clear()

    def _end_turn(self) -> None:
        """End current turn: STT -> LLM -> TTS."""
        self._turn_count += 1

        audio_samples = self._speech_samples if self._speech_samples else self._audio_buffer
        if not audio_samples:
            return

        audio = (
            np.concatenate(audio_samples)
            if audio_samples
            else np.array([], dtype=np.float32)
        )
        if len(audio) < SAMPLE_RATE * 0.5:
            return

        # STT
        try:
            text = transcribe(audio)
        except Exception as e:
            logger.error(f"STT error: {e}")
            if self._on_error:
                self._on_error(f"STT error: {e}")
            return

        if not text.strip():
            return

        logger.info(f"STT: {text}")

        # Add to messages
        self._messages.append({"role": "user", "content": text})

        # LLM
        try:
            response = generate_stream(
                self._messages,
                self._cerebras_client,
                self._tts_queue,
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            if self._on_error:
                self._on_error(f"LLM error: {e}")
            return

        if not response:
            return

        self._messages.append({"role": "assistant", "content": response})

        # TTS: process queued chunks
        self._process_tts_queue()

    def _process_tts_queue(self) -> None:
        """Process TTS queue and send audio to output."""
        self._tts_processing = True
        try:
            while self._tts_queue:
                chunk = self._tts_queue.pop(0)
                if not chunk.strip():
                    continue
                logger.info(f"TTS: {chunk}")
                try:
                    audio = synthesize(chunk)
                    if len(audio) > 0 and self._on_audio_output:
                        self._on_audio_output(audio)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
        finally:
            self._tts_processing = False

    def interrupt(self) -> None:
        """Request interruption of current TTS."""
        self._interrupted = True

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def turn_count(self) -> int:
        return self._turn_count
