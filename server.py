"""Main server entry point.

Runs an HTTP server that:
1. Serves the phone web UI
2. Handles WebSocket signaling via WebRTCTransport
3. Orchestrates the voice pipeline when a call is active
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).parent.resolve()
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("server")

import config


async def handle_signaling(request: web.Request) -> web.Response:
    """Delegate to WebRTCTransport for signaling handling."""
    transport = request.app.get("transport")
    if transport:
        return await transport.handle_signaling(request)
    return web.Response(status=500, text="No transport configured")


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
    # Load system prompt
    system_prompt_path = ROOT / config.SYSTEM_PROMPT_FILE
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text().strip()
    else:
        system_prompt = "You are a helpful assistant."
    logger.info(f"System prompt loaded ({len(system_prompt)} chars) from {config.SYSTEM_PROMPT_FILE}")

    # Initialize Cerebras client
    try:
        from cerebras.cloud.sdk import Client
        cerebras_client = Client(api_key=config.CEREBRAS_API_KEY)
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
        silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(silence, path_or_hf_repo=config.STT_MODEL)
        logger.info("STT warmed")
    except Exception as e:
        logger.warning(f"STT warm failed: {e}")

    # Create transport, agent, pipeline
    from transport.webrtc import WebRTCTransport
    from agent import VoiceAgent
    from pipeline import Pipeline

    transport = WebRTCTransport()
    agent = VoiceAgent(system_prompt=system_prompt)
    pipeline = Pipeline(agent)

    agent.set_cerebras_client(cerebras_client)

    # Wire transport callbacks to pipeline
    transport.set_audio_received_callback(pipeline.on_audio_received)
    transport.set_barge_in_callback(pipeline.interrupt)

    app = web.Application()
    app["pipeline"] = pipeline
    app["agent"] = agent
    app["transport"] = transport
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
http://{lan_ip}:{config.PORT}

Open that URL on your phone's browser (same Wi-Fi).
""")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", config.PORT)
    await site.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        pipeline.stop()
        agent.stop()
        await transport.close()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)
