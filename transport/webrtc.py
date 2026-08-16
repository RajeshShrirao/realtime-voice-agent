"""
Local WebRTC transport.

Uses aiortc for pure-Python WebRTC without any cloud media provider.
The server acts as a WebRTC peer that receives audio from the browser
and sends synthesized TTS audio back.

Signaling is done over a WebSocket connection on the same HTTP server.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Callable, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import WebSocketAsync

logger = logging.getLogger("webrtc")


class WebRTCTransport:
    """
    Manages a single WebRTC peer connection for audio transport.

    Audio flow:
        Browser mic -> WebRTC -> transport._on_track -> audio_callback
        TTS output -> transport.enqueue_audio -> RTCPeerConnection track
    """

    def __init__(self):
        self._pc: Optional[RTCPeerConnection] = None
        self._audio_callback: Optional[Callable[[np.ndarray], None]] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._done = asyncio.Event()
        self._receiver_task: Optional[asyncio.Task] = None

    async def connect(self, signaling_url: str, audio_callback: Callable[[np.ndarray], None]) -> None:
        """
        Connect to the signaling server and establish a WebRTC peer connection.

        :param signaling_url: WebSocket URL for signaling (e.g. ws://localhost:7860/ws)
        :param audio_callback: Called with each incoming audio frame as a numpy array
        """
        self._audio_callback = audio_callback
        self._done.clear()

        try:
            async with WebSocketAsync(signaling_url) as ws:
                # Wait for offer from browser
                offer = await ws.receive()
                logger.info("Received offer from browser")

                self._pc = RTCPeerConnection()
                self._pc.on_track = self._on_track
                self._pc.on_connectionstatechange = self._on_connection_state

                # Handle incoming audio track
                @self._pc.on_track
                def on_track(track):
                    if track.kind == "audio":
                        self._receiver_task = asyncio.create_task(
                            self._receive_audio(track)
                        )

                # Create answer
                await self._pc.setRemoteDescription(RTCSessionDescription(**offer))
                answer = await self._pc.createAnswer()
                await self._pc.setLocalDescription(answer)

                # Send answer
                await ws.send(self._pc.localDescription.sdp)
                logger.info("Sent answer to browser")

                # Wait for connection to close
                await self._done.wait()

        except Exception as e:
            logger.error(f"WebRTC connection error: {e}")
        finally:
            await self.close()

    def _on_track(self, track) -> None:
        logger.info(f"Track received: {track.kind}")

    def _on_connection_state(self) -> None:
        state = self._pc.connectionState if self._pc else "unknown"
        logger.info(f"Connection state: {state}")
        if state in ("closed", "failed", "disconnected"):
            self._done.set()

    async def _receive_audio(self, track) -> None:
        """Receive audio frames from the browser and pass to the callback."""
        try:
            while not self._done.is_set():
                frame = await track.recv()
                if frame is None:
                    break
                # Convert WebRTC frame to numpy
                audio = np.array(frame.samples, dtype=np.float32)
                if self._audio_callback:
                    self._audio_callback(audio)
        except Exception as e:
            logger.error(f"Audio receive error: {e}")
        finally:
            self._done.set()

    async def enqueue_audio(self, audio: np.ndarray) -> None:
        """Queue audio to be sent to the browser."""
        if self._pc is None or self._pc.connectionState != "connected":
            return
        try:
            await self._audio_queue.put(audio)
        except asyncio.CancelledError:
            pass

    async def _audio_sender(self) -> None:
        """Send queued audio to the browser."""
        # Create an audio track for sending
        if self._pc is None:
            return

        # We use a simple approach: create a media stream and add it
        # For sending, we use a MediaStreamTrack that produces our audio
        from aiortc import MediaStreamTrack
        from aiortc._media import AudioFrame

        class AudioSenderTrack(MediaStreamTrack):
            kind = "audio"

            def __init__(self, queue):
                super().__init__()
                self._queue = queue
                self._start = 0

            async def recv(self):
                try:
                    audio = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Send silence if no audio available
                    import ctypes
                    samples = (ctypes.c_float * 160)(*([0.0] * 160))
                    return AudioFrame(
                        kind="audio",
                        samples=samples,
                        sample_rate=TTS_SAMPLE_RATE,
                        layout="mono",
                        timestamp=self._start,
                    )
                self._start += len(audio)
                # Convert numpy to ctypes array for AudioFrame
                import ctypes
                samples_type = ctypes.c_float * len(audio)
                samples = samples_type(*audio.astype(np.float32).tolist())
                return AudioFrame(
                    kind="audio",
                    samples=samples,
                    sample_rate=TTS_SAMPLE_RATE,
                    layout="mono",
                    timestamp=self._start,
                )

        sender_track = AudioSenderTrack(self._audio_queue)
        self._pc.addTrack(sender_track)

    async def close(self) -> None:
        """Close the WebRTC connection."""
        self._done.set()
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
        if self._pc:
            await self._pc.close()
            self._pc = None
        logger.info("WebRTC connection closed")

    @property
    def is_connected(self) -> bool:
        return self._pc is not None and self._pc.connectionState == "connected"
