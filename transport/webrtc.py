"""
Local WebRTC transport for the Realtime Voice Agent.

Uses aiortc for pure-Python WebRTC without any cloud media provider.
The server acts as a WebRTC peer that receives audio from the browser
and sends synthesized TTS audio back.

Signaling is done over a WebSocket connection on the same HTTP server.

Usage:
    transport = WebRTCTransport()
    await transport.handle_signaling(request, pipeline)
"""

import asyncio
import json
import logging
from ctypes import c_float
from pathlib import Path
from typing import Optional

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioFrame, MediaStreamTrack

logger = logging.getLogger("webrtc")


def audioframe_to_numpy(frame: AudioFrame) -> np.ndarray:
    """Convert an aiortc AudioFrame to a 1D numpy float32 array."""
    arr = frame.to_ndarray()
    return arr.flatten().astype(np.float32)


class AudioSenderTrack(MediaStreamTrack):
    """MediaStreamTrack that sends queued TTS audio to the browser."""

    kind = "audio"

    def __init__(self, sample_rate: int, queue: asyncio.Queue):
        super().__init__()
        self._sample_rate = sample_rate
        self._queue = queue
        self._timestamp = 0
        self._frame_count = 0

    async def recv(self) -> Optional[AudioFrame]:
        try:
            audio = await asyncio.wait_for(self._queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            audio = None

        if audio is None or len(audio) == 0:
            silence = np.zeros((1, 240), dtype=np.float32)
            frame = AudioFrame.from_ndarray(silence, format="fltp", layout="mono")
            frame.sample_rate = self._sample_rate
            frame.pts = self._timestamp
            self._timestamp += 240
            self._frame_count += 1
            return frame

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio_2d = audio.reshape(1, -1)
        else:
            audio_2d = audio

        frame = AudioFrame.from_ndarray(audio_2d, format="fltp", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        self._timestamp += len(audio_2d[0])
        self._frame_count += 1
        return frame


class WebRTCTransport:
    """
    Manages a single WebRTC peer connection for audio transport.

    Audio flow:
        Browser mic → WebRTC → transport._on_track → audio_callback
        TTS output → audio_queue → AudioSenderTrack → browser
    """

    def __init__(self):
        self._pc: Optional[RTCPeerConnection] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._sender_track: Optional[AudioSenderTrack] = None
        self._call_active = False
        self._on_audio_received = None
        self._on_started = None
        self._on_stopped = None
        self._on_barge_in = None

    def set_audio_received_callback(self, callback) -> None:
        """Set callback for incoming audio frames from browser."""
        self._on_audio_received = callback

    def set_started_callback(self, callback) -> None:
        """Set callback when WebRTC connection is established."""
        self._on_started = callback

    def set_stopped_callback(self, callback) -> None:
        """Set callback when WebRTC connection is closed."""
        self._on_stopped = callback

    def set_barge_in_callback(self, callback) -> None:
        """Set callback for barge-in signals from browser."""
        self._on_barge_in = callback

    def get_audio_queue(self) -> asyncio.Queue:
        """Return the audio queue for piping TTS output to the browser."""
        if self._audio_queue is None:
            self._audio_queue = asyncio.Queue()
        return self._audio_queue

    async def handle_signaling(self, request: web.Request) -> web.Response:
        """WebSocket handler for WebRTC signaling."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        logger.info("WebSocket connection opened")

        pipeline_ref = request.app.get("pipeline")
        agent_ref = request.app.get("agent")

        async def cleanup():
            self._call_active = False
            if self._sender_track and self._audio_queue:
                try:
                    self._audio_queue.put_nowait(None)
                except Exception:
                    pass
            if self._pc:
                await self._pc.close()
            await ws.close()
            if pipeline_ref:
                pipeline_ref.stop()
            if agent_ref:
                agent_ref.stop()
            logger.info("Call cleanup complete")

        async for msg in ws:
            try:
                data = json.loads(msg.data)
                action = data.get("action")

                if action == "offer":
                    await self._handle_offer(ws, data, pipeline_ref, agent_ref)

                elif action == "end_call":
                    logger.info("End call requested by browser")
                    await cleanup()

                elif action == "barge_in":
                    logger.info("Barge-in detected from browser")
                    if self._on_barge_in:
                        self._on_barge_in()

            except json.JSONDecodeError:
                logger.warning("Invalid JSON from client")
            except Exception as e:
                logger.error(f"Signaling error: {e}")
                try:
                    await ws.send_json({"action": "error", "message": str(e)})
                except Exception:
                    pass

        logger.info("WebSocket connection closed")
        await cleanup()
        return ws

    async def _handle_offer(
        self,
        ws: web.WebSocketResponse,
        data: dict,
        pipeline_ref,
        agent_ref,
    ) -> None:
        """Handle a WebRTC offer from the browser."""
        logger.info("Received offer from browser")

        self._pc = RTCPeerConnection()
        self._audio_queue = asyncio.Queue()
        self._sender_track = AudioSenderTrack(24000, self._audio_queue)
        self._pc.addTrack(self._sender_track)

        offer_sdp = data.get("sdp")
        offer_type = data.get("type", "offer")

        await self._pc.setRemoteDescription(RTCSessionDescription(offer_sdp, offer_type))

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        await ws.send_json({
            "action": "answer",
            "sdp": self._pc.localDescription.sdp,
            "type": self._pc.localDescription.type,
        })
        logger.info("Sent answer to browser")
        self._call_active = True

        # ICE connected → start pipeline
        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            state = self._pc.iceConnectionState
            logger.info(f"ICE state: {state}")
            if state == "connected":
                logger.info("WebRTC connected — starting pipeline")
                if pipeline_ref:
                    pipeline_ref.start()
                if pipeline_ref:
                    pipeline_ref.set_audio_sender_queue(self._audio_queue)
                if self._on_started:
                    self._on_started()

        @self._pc.on("connectionstatechange")
        async def on_connection_state():
            state = self._pc.connectionState
            logger.info(f"Connection state: {state}")
            if state in ("closed", "failed", "disconnected"):
                self._call_active = False
                if self._on_stopped:
                    self._on_stopped()
                if pipeline_ref:
                    pipeline_ref.stop()
                if agent_ref:
                    agent_ref.stop()
                await self._pc.close()

        @self._pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info("Audio track received from browser")

                async def receive_audio():
                    try:
                        while True:
                            frame = await track.recv()
                            if frame is None:
                                break
                            audio = audioframe_to_numpy(frame)
                            if self._on_audio_received and self._call_active:
                                self._on_audio_received(audio)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Audio receive error: {e}")

                asyncio.create_task(receive_audio())

    async def send_audio(self, audio: np.ndarray) -> None:
        """Queue audio to be sent to the browser."""
        if self._audio_queue is None:
            return
        try:
            self._audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping frame")
        except RuntimeError:
            pass

    async def close(self) -> None:
        """Close the WebRTC connection."""
        self._call_active = False
        if self._sender_track and self._audio_queue:
            try:
                self._audio_queue.put_nowait(None)
            except Exception:
                pass
        if self._pc:
            await self._pc.close()
            self._pc = None
        logger.info("WebRTC connection closed")

    @property
    def is_connected(self) -> bool:
        return self._pc is not None and self._pc.connectionState == "connected"
