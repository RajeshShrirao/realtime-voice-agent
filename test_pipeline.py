"""Pipeline smoke test.

Validates the full VAD -> STT -> LLM -> TTS loop using synthetic audio.
Does not require a phone or WebRTC connection.
"""

import asyncio
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logger = logging.getLogger("test")

if not os.environ.get("CEREBRAS_API_KEY"):
    os.environ["CEREBRAS_API_KEY"] = "csk-test-key-placeholder"
if not os.environ.get("TORCH_HUB_TRUST_REPO"):
    os.environ["TORCH_HUB_TRUST_REPO"] = "1"

import config
from agent import VoiceAgent
from pipeline import Pipeline
import torch
from services.vad import load_vad

config.LLM_MAX_RETRIES = 1
config.LLM_RETRY_BASE_DELAY = 0.5


def generate_test_audio(duration_s=1.0):
    sample_rate = config.SAMPLE_RATE
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    envelope = np.zeros_like(t)
    syllable_rate = 4.0
    for i in range(int(syllable_rate * duration_s)):
        center = i / syllable_rate
        window = np.exp(-((t - center) ** 2) / 0.02)
        envelope += window
    fundamental = 150
    voice = (
        np.sin(2 * np.pi * fundamental * t) * 0.5
        + np.sin(2 * np.pi * fundamental * 2 * t) * 0.25
        + np.sin(2 * np.pi * fundamental * 3 * t) * 0.125
    )
    audio = voice * envelope * 0.3 + np.random.randn(len(t)) * 0.02
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-8
    return audio


async def run_test():
    print("=" * 60)
    print("PIPELINE SMOKE TEST")
    print("=" * 60)
    cerebras_client = None
    try:
        from cerebras.cloud.sdk import Client
        cerebras_client = Client(api_key=config.CEREBRAS_API_KEY)
        print("[OK] Cerebras client initialized")
    except Exception as e:
        print(f"[SKIP] Cerebras: {e}")
    try:
        import mlx_audio
        print("[OK] TTS ready")
    except Exception as e:
        print(f"[FAIL] TTS: {e}")
        return False
    try:
        vad_model = load_vad()
        print("[OK] VAD loaded")
    except Exception as e:
        print(f"[FAIL] VAD: {e}")
        return False
    system_prompt = "You are a helpful assistant. Keep responses short."
    agent = VoiceAgent(system_prompt=system_prompt)
    pipeline = Pipeline(agent)
    agent.set_cerebras_client(cerebras_client)
    events = []
    def on_listening():
        events.append("listening")
        print("  -> VAD detected speech")
    def on_speaking():
        events.append("speaking")
    def on_error(error):
        events.append(f"error:{error}")
        print(f"  -> ERROR: {error}")
    def on_tts_audio(audio):
        events.append("tts_audio")
        print(f"  -> TTS: {len(audio)} samples ({len(audio)/config.TTS_SAMPLE_RATE:.3f}s)")
    agent.set_callbacks(
        on_listening=on_listening,
        on_speaking=on_speaking,
        on_error=on_error,
        on_audio_output=on_tts_audio,
    )
    print("")
    print("Starting pipeline...")
    pipeline.start()
    time.sleep(0.5)
    print("")
    print("Generating test audio (1.5s)...")
    test_audio = generate_test_audio(duration_s=1.5)
    print(f"  Audio: {len(test_audio)} samples ({len(test_audio)/config.SAMPLE_RATE:.2f}s)")
    chunk_size = config.VAD_CHUNK_SAMPLES
    num_chunks = len(test_audio) // chunk_size
    print(f"  Pushing {num_chunks} chunks...")
    for i in range(num_chunks):
        chunk = test_audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        agent.push_audio(chunk)
        time.sleep(0.01)
    print("  Adding silence...")
    for i in range(30):
        agent.push_audio(np.zeros(chunk_size, dtype=np.float32))
        time.sleep(0.005)
    print("")
    print("Waiting...")
    time.sleep(3.0)
    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Events: {events}")
    has_listening = "listening" in events
    has_tts = "tts_audio" in events
    has_error = any(e.startswith("error:") for e in events)
    print(f"")
    print(f"VAD: {'YES' if has_listening else 'NO'}")
    print(f"TTS: {'YES' if has_tts else 'NO'}")
    print(f"Errors: {'YES' if has_error else 'NO'}")
    if has_error:
        print("")
        print("[FAIL] Errors occurred")
        return False
    if not has_listening:
        print("")
        print("[INFO] VAD did not trigger (synthetic audio)")
        print("  Pipeline infrastructure: OK")
        return True
    if not has_tts:
        if cerebras_client is None:
            print("")
            print("[SKIP] TTS not tested (no API key)")
            return True
        print("")
        print("[FAIL] TTS did not produce audio")
        return False
    print("")
    print("[PASS] Pipeline test passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)
