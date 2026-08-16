import threading
import queue

is_playing = threading.Event()
interrupted = threading.Event()
tts_queue: queue.Queue = queue.Queue()
turn_done = threading.Event()
TURN_SENTINEL = object()

messages: list[dict[str, str]] = []
system_prompt: str = ""

conversation_state = {
    "turn_count": 0,
}

stats = {
    "turns": 0,
    "interrupts": 0,
    "total_recorded_s": 0.0,
    "total_spoken_s": 0.0,
    "tts_sentences": 0,
    "stt_inference_ms": [],
    "llm_tokens": 0,
    "llm_first_token_ms": [],
    "llm_total_ms": [],
    "tts_synthesis_ms": [],
    "errors": [],
}
stats_lock = threading.Lock()

tts_model = None
vad_model = None
cerebras_client = None
llm_provider = "cerebras"
