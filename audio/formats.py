# Audio format constants for the voice pipeline.

SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
VAD_CHUNK_SAMPLES = 512
PRE_ROLL_CHUNKS = 5
FADE_SAMPLES = 480
GAP_SAMPLES = int(TTS_SAMPLE_RATE * 0.025)  # 25ms gap between TTS chunks
BARGE_IN_FRAMES = 8          # VAD frames before interrupting
BARGE_IN_THRESHOLD_S = 0.250  # 250ms barge-in threshold
