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

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

ROOT = Path(__file__).parent.resolve()
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("server")

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY environment variable is required")

PORT = int(os.environ.get("PORT", "7860"))

# Config constants
SAMPLE_RATE = 16000
STT_MODEL = "mlx-community/whisper-small-mlx"
TTS_MODEL = "mlx-community/Chatterbox-Turbo-TTS-4bit"
LLM_MODEL_ID = "qwen-3-235b-a22b-instruct-2507"


def load_system_prompt() -> str:
    prompt_path = ROOT / "prompts" / "system.txt"
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    return "You are a helpful assistant."


async def handle_signaling(request: web.Request) -> web.Response:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info("WebSocket connection opened")

    pc = RTCPeerConnection()
    pipeline_ref = request.app.get("pipeline")
    agent_ref = request.app.get("agent")
    call_active = False

    async for msg in ws:
        try:
            data = json.loads(msg.data)
            action = data.get("action")

            if action == "offer":
                logger.info("Received offer")
                offer_sdp = data.get("sdp")
                offer_type = data.get("type", "offer")

                await pc.setRemoteDescription(RTCSessionDescription(offer_sdp, offer_type))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await ws.send_json({
                    "action": "answer",
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                })
                logger.info("Sent answer")
                call_active = True

                @pc.on("iceconnectionstatechange")
                async def on_ice_state():
                    state = pc.iceConnectionState
                    logger.info(f"ICE state: {state}")
                    if state == "connected":
                        logger.info("WebRTC connected")
                        if pipeline_ref:
                            pipeline_ref.start()
                        await ws.send_json({"action": "connected"})

                @pc.on("connectionstatechange")
                async def on_connection_state():
                    state = pc.connectionState
                    logger.info(f"Connection state: {state}")
                    if state in ("closed", "failed", "disconnected"):
                        logger.info("WebRTC disconnected")
                        if pipeline_ref:
                            pipeline_ref.stop()
                        if agent_ref:
                            agent_ref.stop()
                        await ws.close()

                @pc.on("track")
                def on_track(track):
                    if track.kind == "audio":
                        logger.info("Audio track received")

                        async def receive_audio():
                            try:
                                while True:
                                    frame = await track.recv()
                                    if frame is None:
                                        break
                                    import numpy as np
                                    audio = np.array(frame.samples, dtype=np.float32)
                                    if pipeline_ref:
                                        pipeline_ref.on_audio_received(audio)
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error(f"Audio receive error: {e}")

                        asyncio.create_task(receive_audio())

            elif action == "end_call":
                logger.info("End call requested")
                call_active = False
                if pipeline_ref:
                    pipeline_ref.stop()
                if agent_ref:
                    agent_ref.stop()
                await pc.close()
                await ws.close()
                break

        except json.JSONDecodeError:
            logger.warning("Invalid JSON from client")
        except Exception as e:
            logger.error(f"Signaling error: {e}")
            try:
                await ws.send_json({"action": "error", "message": str(e)})
            except Exception:
                pass

    logger.info("WebSocket connection closed")
    if call_active:
        if pipeline_ref:
            pipeline_ref.stop()
        if agent_ref:
            agent_ref.stop()
    await pc.close()
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
        import numpy as np
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
