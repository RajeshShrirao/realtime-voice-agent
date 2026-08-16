import os

import numpy as np

import config
import state
import utils


def synthesize(text):
    if config.TEXT_MODE or state.tts_model is None:
        return np.array([], dtype=np.float32)

    text = text.strip().rstrip(".")
    if not text:
        return np.array([], dtype=np.float32)
    if not any(c.isalpha() for c in text):
        return np.array([], dtype=np.float32)
    try:
        with utils.measure_time("tts_synthesis_ms"):
            with utils.Silence():
                results = list(
                    state.tts_model.generate(
                        text,
                        temperature=config.TTS_TEMPERATURE,
                        ref_audio=None,
                        language="en",
                    )
                )
        utils.record_stat("tts_sentences", increment=1)
        if results and results[0].audio is not None:
            return np.array(results[0].audio, dtype=np.float32)
    except Exception as e:
        utils.record_stat("errors", f"TTS: {e}")
    return np.array([], dtype=np.float32)
