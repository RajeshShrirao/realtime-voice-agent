import numpy as np

import mlx_whisper
import config
import utils


def transcribe(audio):
    if len(audio) < config.SAMPLE_RATE * 0.5:
        return ""
    with utils.measure_time("stt_inference_ms"):
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=config.STT_MODEL,
            language="en",
        )
    if result.get("no_speech_prob", 0) > 0.6:
        return ""
    return result.get("text", "").strip()


def warm_stt():
    if config.TEXT_MODE:
        return
    with utils.Silence():
        mlx_whisper.transcribe(
            np.zeros(config.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=config.STT_MODEL,
        )
