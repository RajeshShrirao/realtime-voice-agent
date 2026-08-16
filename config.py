import os
import sys

TEXT_MODE = "--text" in sys.argv

# Audio
SAMPLE_RATE = 16000
VAD_CHUNK = 512
PRE_ROLL = 5
_FADE_SAMPLES = 480
_TTS_SR = 24000
_GAP_SAMPLES = int(_TTS_SR * 0.025)

# LLM
LLM_MODEL_ID = "qwen-3-235b-a22b-instruct-2507"
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY environment variable is required")

LLM_TEMPERATURE = 0.8
LLM_TOP_P = 0.88
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 2

# STT
STT_MODEL = "mlx-community/whisper-small-mlx"

# TTS
TTS_MODEL = "mlx-community/Chatterbox-Turbo-TTS-4bit"
