# CLAUDE.md

This file provides guidance when working with the **realtime-voice-agent** repository.

## Project Overview

**Realtime Voice Agent** — a local realtime voice-agent foundation running on Apple Silicon. Realtime conversational pipeline: VAD → STT → LLM → TTS, with barge-in support. LLM runs on Cerebras Cloud (qwen-3-235b) for sub-second latency.

The project is designed as a reusable engine. The transport layer (currently local WebRTC) can be replaced with telephony providers without rewriting the AI pipeline.

## Running the Project

```bash
source .venv/bin/activate
python3 server.py
```

- **Requirements:** Python 3.10+, Apple Silicon (M-series).
- **LLM:** Cerebras Cloud API. API key must be set via `CEREBRAS_API_KEY` env var.
- **Core Dependencies:** `torch` (Silero VAD), `mlx-whisper`, `mlx-audio`, `sounddevice`, `cerebras-cloud-sdk`.

## Architecture

| File | Purpose |
|------|---------|
| `server.py` | Main entry. HTTP server + WebRTC signaling + voice pipeline orchestration. |
| `agent.py` | VoiceAgent abstraction: system prompt, VAD, STT, LLM, TTS, transport. |
| `pipeline.py` | Realtime voice pipeline wiring. |
| `audio/input.py` | Audio input handling (WebRTC frames → VAD). |
| `audio/output.py` | Audio output handling (TTS → WebRTC). |
| `audio/formats.py` | Audio format constants and conversion. |
| `services/vad.py` | Silero VAD wrapper. |
| `services/stt.py` | mlx-whisper STT wrapper. |
| `services/llm.py` | Cerebras LLM client with streaming and retry. |
| `services/tts.py` | Chatterbox TTS wrapper. |
| `transport/webrtc.py` | Local WebRTC transport. |
| `config.py` | All constants and configuration. |
| `state.py` | Shared mutable state (thread-safe globals). |
| `utils.py` | Misc utilities (timing, stats). |
| `web/index.html` | Phone browser UI. |
| `web/app.js` | Phone browser logic. |
| `web/style.css` | Phone browser styles. |
| `prompts/system.txt` | Configurable system prompt. |

### Pipeline Flow

1.  **VAD:** Silero VAD detects speech in incoming audio frames.
2.  **STT:** `mlx-whisper-small` transcribes buffered audio.
3.  **LLM:** `qwen-3-235b-a22b-instruct-2507` via Cerebras Cloud. Streaming text with temp=0.8, top_p=0.88. Strips `<|channel>...<channel|>` thinking tags. Auto-retries on 429 rate-limit (3 attempts, exponential backoff).
4.  **TTS:** `Chatterbox-Turbo-TTS-4bit` synthesizes speech. Sentence-chunked streaming.
5.  **Barge-in:** Parallel playback monitor detects user speech during TTS and interrupts.

### Key Config Constants

```python
SAMPLE_RATE = 16000
LLM_MODEL_ID = "qwen-3-235b-a22b-instruct-2507"
TTS_MODEL = "mlx-community/Chatterbox-Turbo-TTS-4bit"
STT_MODEL = "mlx-community/whisper-small-mlx"
LLM_TEMPERATURE = 0.8
LLM_TOP_P = 0.88
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 2
```

## Development Guidelines

- **No LM Studio dependency:** LLM runs on Cerebras Cloud. No local server needed.
- **MLX Limited:** Only STT (`mlx_whisper`) and TTS (`mlx_audio`) use MLX. LLM is cloud API.
- **Async/Threading:** Voice and TTS run in separate threads. Be careful with shared state.
- **Transport abstraction:** The VoiceAgent should not know it's talking to a phone. Transport must be replaceable.
- **Barge-in:** Ensure `is_playing` and `interrupted` events are correct in new audio features.
- **Thinking Tokens:** Model outputs thinking in `<|channel>...<channel|>` format. Stripped before TTS.
- **No persistent memory in V1:** Only current conversation in process memory. No SQLite, embeddings, or vector DB.
- **No authentication in V1:** Local-network-only. Document this clearly.
- **API keys:** Never send to the browser. Only the Mac server communicates with Cerebras.
