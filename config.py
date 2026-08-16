"""Configuration for the Realtime Voice Agent.

All settings can be overridden via environment variables.
See .env.example for available options.
"""

import os
import sys

# ============================================================
# Mode
# ============================================================
TEXT_MODE = "--text" in sys.argv

# ============================================================
# Server
# ============================================================
PORT = int(os.environ.get("PORT", "7860"))

# ============================================================
# LLM — Cerebras Cloud
# ============================================================
LLM_PROVIDER = "cerebras"
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen-3-235b-a22b-instruct-2507")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.8"))
LLM_TOP_P = float(os.environ.get("LLM_TOP_P", "0.88"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_BASE_DELAY", "2"))

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY environment variable is required")

# ============================================================
# STT — mlx-whisper
# ============================================================
STT_MODEL = os.environ.get("STT_MODEL", "mlx-community/whisper-small-mlx")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")
STT_NO_SPEECH_THRESHOLD = float(os.environ.get("STT_NO_SPEECH_THRESHOLD", "0.6"))

# ============================================================
# TTS — Chatterbox-Turbo
# ============================================================
TTS_MODEL = os.environ.get("TTS_MODEL", "mlx-community/Chatterbox-Turbo-TTS-4bit")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
TTS_TEMPERATURE = float(os.environ.get("TTS_TEMPERATURE", "0.6"))

# ============================================================
# Audio
# ============================================================
SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512
PRE_ROLL_CHUNKS = 5
FADE_SAMPLES = 480
GAP_SAMPLES = int(TTS_SAMPLE_RATE * 0.025)  # 25ms between TTS chunks

# VAD settings
VAD_SPeech_PROB_THRESHOLD = float(os.environ.get("VAD_SPEECH_PROB_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_DURATION_S = float(os.environ.get("VAD_MIN_SPEECH_DURATION_S", "0.4"))
VAD_MAX_SILENCE_DURATION_S = float(os.environ.get("VAD_MAX_SILENCE_DURATION_S", "2.4"))

# Barge-in
BARGE_IN_THRESHOLD_S = float(os.environ.get("BARGE_IN_THRESHOLD_S", "0.250"))
BARGE_IN_VAD_FRAMES = int(os.environ.get("BARGE_IN_VAD_FRAMES", "8"))

# ============================================================
# Agent behavior
# ============================================================
SYSTEM_PROMPT_FILE = os.environ.get("SYSTEM_PROMPT_FILE", "prompts/system.txt")
MAX_RESPONSE_SENTENCES = int(os.environ.get("MAX_RESPONSE_SENTENCES", "3"))
RESPONSE_TIMEOUT_S = float(os.environ.get("RESPONSE_TIMEOUT_S", "15"))
