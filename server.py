"""Main server entry point.

Runs an HTTP server that:
1. Serves the phone web UI
2. Handles WebSocket signaling for WebRTC
3. Orchestrates the voice pipeline when a call is active
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioFrame, MediaStreamTrack

ROOT = Path(__file__).parent.resolve()
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("server")

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY environment variable is required")

PORT = int(os.environ.get("PORT", "7860"))

SAMPLE_RATE = 16000
STT_MODEL = "mlx-community/whisper-small-mlx"
TTS_SAMPLE_RATE = 24000
LLM_MODEL_ID = "qwen-3-235b-a22b-instruct-2507"


def load_system_prompt() -> str:
    prompt_path = ROOT / "prompts" / "system.txt"
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    return "You are a helpful assistant."


def audioframe_to_numpy(frame: AudioFrame) -> np.ndarray:
    """Convert an aiortc AudioFrame to a 1D numpy float32 array."""
    arr = frame.to_ndarray()  # shape (channels, samples)
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
            # Send silence: 160 samples (10ms at 16kHz, but we use 24kHz TTS sr)
            silence = np.zeros((1, 240), dtype=np.float32)  # 10ms at 24kHz
            frame = AudioFrame.from_ndarray(silence, format="fltp", layout="mono")
            frame.sample_rate = self._sample_rate
            frame.pts = self._timestamp
            self._timestamp += 240
            self._frame_count += 1
            return frame

        # Convert 1D numpy to 2D for AudioFrame (channels, samples)
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


async def handle_signaling(request: web.Request) -> web.Response:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info("WebSocket connection opened")

    pc: Optional[RTCPeerConnection] = None
    pipeline_ref = request.app.get("pipeline")
    agent_ref = request.app.get("agent")
    call_active = False
    audio_queue: asyncio.Queue = asyncio.Queue()
    sender_track: Optional[AudioSenderTrack] = None

    async def cleanup():
        nonlocal pc, sender_track, call_active
        call_active = False
        if sender_track:
            try:
                sender_track._queue.put_nowait(None)
            except Exception:
                pass
        if pc:
            await pc.close()
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
                logger.info("Received offer from browser")
                offer_sdp = data.get("sdp")
                offer_type = data.get("type", "offer")

                pc = RTCPeerConnection()
                await pc.setRemoteDescription(RTCSessionDescription(offer_sdp, offer_type))

                # Create audio sender track and add to PC
                sender_track = AudioSenderTrack(TTS_SAMPLE_RATE, audio_queue)
                pc.addTrack(sender_track)

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await ws.send_json({
                    "action": "answer",
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                })
                logger.info("Sent answer to browser")
                call_active = True

                # ICE connected → start pipeline
                @pc.on("iceconnectionstatechange")
                async def on_ice_state():
                    state = pc.iceConnectionState
                    logger.info(f"ICE state: {state}")
                    if state == "connected":
                        logger.info("WebRTC connected — starting pipeline")
                        if pipeline_ref:
                            pipeline_ref.start()
                        if pipeline_ref:
                            pipeline_ref.set_audio_sender_queue(audio_queue)
                        await ws.send_json({"action": "connected"})

                @pc.on("connectionstatechange")
                async def on_connection_state():
                    state = pc.connectionState
                    logger.info(f"Connection state: {state}")
                    if state in ("closed", "failed", "disconnected"):
                        await cleanup()

                @pc.on("track")
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
                                    if pipeline_ref and call_active:
                                        pipeline_ref.on_audio_received(audio)
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error(f"Audio receive error: {e}")

                        asyncio.create_task(receive_audio())

            elif action == "end_call":
                logger.info("End call requested by browser")
                await cleanup()

            elif action == "barge_in":
                logger.info("Barge-in detected from browser")
                if pipeline_ref:
                    pipeline_ref.interrupt()
                if agent_ref:
                    agent_ref.interrupt()

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


def get_lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


async def start_server():
    system_prompt = load_system_prompt()
    logger.info(f"System prompt loaded ({len(system_prompt)} chars)")

    # Initialize Cerebras client
    try:
        from cerebras.cloud.sdk import Client
        cerebras_client = Client(api_key=CEREBRAS_API_KEY)
        logger.info("Cerebras client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Cerebras client: {e}")
        raise

    # Initialize TTS model
    try:
        import mlx_audio
        logger.info("TTS ready (mlx_audio)")
    except Exception as e:
        logger.error(f"Failed to load TTS: {e}")
        raise

    # Warm STT
    try:
        import mlx_whisper
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(silence, path_or_hf_repo=STT_MODEL)
        logger.info("STT warmed")
    except Exception as e:
        logger.warning(f"STT warm failed: {e}")

    # Create agent and pipeline
    from agent import VoiceAgent
    from pipeline import Pipeline

    agent = VoiceAgent(system_prompt=system_prompt)
    pipeline = Pipeline(agent)
    agent.set_cerebras_client(cerebras_client)

    app = web.Application()
    app["pipeline"] = pipeline
    app["agent"] = agent
    app["cerebras_client"] = cerebras_client

    # Serve static files
    web_dir = ROOT / "web"
    app.router.add_get("/", lambda r: web.FileResponse(web_dir / "index.html"))
    app.router.add_get("/app.js", lambda r: web.FileResponse(web_dir / "app.js"))
    app.router.add_get("/style.css", lambda r: web.FileResponse(web_dir / "style.css"))
    app.router.add_get("/ws", handle_signaling)

    lan_ip = get_lan_ip()
    print(f"""
Realtime Voice Agent
────────────────────

🟢 Server running
http://{lan_ip}:{PORT}

Open that URL on your phone's browser (same Wi-Fi).
""")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        pipeline.stop()
        agent.stop()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)
