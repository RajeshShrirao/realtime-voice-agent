# Realtime Voice Agent

> A local realtime voice-agent foundation running on Apple Silicon. Voice pipeline: **VAD → STT → LLM → TTS**, with barge-in support and a local WebRTC phone interface.

---

## Features

- **Sub-second response latency** — LLM runs on [Cerebras Cloud](https://cerebras.ai) (Qwen 3 235B), delivering ~0.5–1s TTFT
- **Real-time voice pipeline** — Silero VAD → mlx-whisper STT → streaming LLM → Chatterbox-Turbo TTS
- **Barge-in support** — Interrupts TTS when the user starts speaking mid-response
- **Local WebRTC phone** — Open a URL on your phone's browser (same Wi-Fi) and start a voice call
- **Streaming LLM + chunked TTS** — First audio plays before the full response is generated
- **Configurable system prompt** — Change the agent's behavior by editing a text file
- **Transport abstraction** — The AI pipeline does not know about the transport. Replace WebRTC with telephony later.

---

## Architecture

```
Phone Browser (WebRTC)
    │
    ▼
Voice Transport (local WebRTC)
    │
    ▼
VAD (Silero)
    │
    ▼
STT (mlx-whisper)
    │
    ▼
LLM (Cerebras Cloud)
    │
    ▼
TTS (Chatterbox-Turbo)
    │
    ▼
Voice Transport (local WebRTC)
    │
    ▼
Phone Speaker
```

### Pipeline Steps

| Step | Component | Detail |
|------|-----------|--------|
| **1. Transport** | Local WebRTC | Receives audio frames from phone browser |
| **2. VAD** | Silero VAD | Detects speech in incoming audio |
| **3. STT** | `mlx-community/whisper-small-mlx` | Transcribes speech to text |
| **4. LLM** | `qwen-3-235b-a22b-instruct-2507` via Cerebras | Streaming response with thinking-token stripping and 429 retry |
| **5. TTS** | `mlx-community/Chatterbox-Turbo-TTS-4bit` | Sentence-chunked synthesis |
| **6. Transport** | Local WebRTC | Sends audio to phone speaker |

---

## Project Structure

```
realtime-voice-agent/
├── server.py            # Main entry — HTTP server + WebRTC + voice pipeline
├── agent.py             # VoiceAgent abstraction
├── pipeline.py          # Pipeline wiring
├── audio/
│   ├── input.py         # Audio input (WebRTC → VAD)
│   ├── output.py        # Audio output (TTS → WebRTC)
│   └── formats.py       # Audio format constants
├── services/
│   ├── vad.py           # Silero VAD wrapper
│   ├── stt.py           # Whisper STT wrapper
│   ├── llm.py           # Cerebras LLM client
│   └── tts.py           # Chatterbox TTS wrapper
├── transport/
│   └── webrtc.py        # Local WebRTC transport
├── config.py            # Configuration constants
├── state.py             # Shared thread-safe state
├── utils.py             # Utilities (timing, stats)
├── web/
│   ├── index.html       # Phone browser UI
│   ├── app.js           # Phone browser logic
│   └── style.css        # Phone browser styles
├── prompts/
│   └── system.txt       # Configurable system prompt
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- A [Cerebras Cloud](https://cerebras.ai) API key

### Installation

```bash
cd realtime-voice-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API Keys

Set your Cerebras API key as an environment variable:

```bash
export CEREBRAS_API_KEY="csk-..."
```

This is required. The server will refuse to start without it.

### Running

```bash
python3 server.py
```

The server will print the LAN URL. Example:

```
Realtime Voice Agent
────────────────────

🟢 Server running
http://192.168.1.42:7860
```

### Phone Connection

1. Ensure your Mac and phone are on the **same Wi-Fi network**.
2. Open the printed URL on your phone's browser.
3. Press **Start Call**.
4. Speak naturally. The agent responds through the phone speaker.

---

## Configuration

All settings are in `config.py` and can be overridden via environment variables.
See `.env.example` for all available options.

### Required

| Variable | Description |
|----------|-------------|
| `CEREBRAS_API_KEY` | Your Cerebras Cloud API key |

### Optional — LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_ID` | `qwen-3-235b-a22b-instruct-2507` | Cerebras model |
| `LLM_TEMPERATURE` | `0.8` | Sampling temperature |
| `LLM_TOP_P` | `0.88` | Top-p sampling |
| `LLM_MAX_TOKENS` | `512` | Max tokens per response |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on 429 |
| `LLM_RETRY_BASE_DELAY` | `2` | Base delay for backoff |

### Optional — STT

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL` | `mlx-community/whisper-small-mlx` | Whisper model |
| `STT_LANGUAGE` | `en` | Language code |
| `STT_NO_SPEECH_THRESHOLD` | `0.6` | No-speech probability threshold |

### Optional — TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODEL` | `mlx-community/Chatterbox-Turbo-TTS-4bit` | TTS model |
| `TTS_TEMPERATURE` | `0.6` | TTS sampling temperature |
| `TTS_SAMPLE_RATE` | `24000` | TTS output sample rate |

### Optional — VAD

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_SPEECH_PROB_THRESHOLD` | `0.5` | Speech probability threshold |
| `VAD_MIN_SPEECH_DURATION_S` | `0.4` | Minimum speech before trigger |
| `VAD_MAX_SILENCE_DURATION_S` | `2.4` | Max silence before end-of-speech |

### Optional — Barge-in

| Variable | Default | Description |
|----------|---------|-------------|
| `BARGE_IN_THRESHOLD_S` | `0.250` | Barge-in threshold in seconds |
| `BARGE_IN_VAD_FRAMES` | `8` | VAD frames before interrupting |

### Optional — Server

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | HTTP server port |

### Optional — Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `SYSTEM_PROMPT_FILE` | `prompts/system.txt` | System prompt file |
| `MAX_RESPONSE_SENTENCES` | `3` | Max sentences per response |
| `RESPONSE_TIMEOUT_S` | `15` | Response timeout |

### System prompt

Edit `prompts/system.txt` to change the agent's behavior.

Example:

```
You are a helpful customer support assistant for Acme Dental.

Answer questions about appointments, insurance, and procedures.
Keep responses concise and friendly.
```

No code changes needed — just edit the text file and restart.

---

## Security

- **V1 is local-network-only.** Anyone on the same Wi-Fi who knows the URL can connect.
- **API keys never leave the Mac.** The browser only communicates with the local server. The server communicates with Cerebras.
- **No authentication in V1.** Do not expose this to the public internet.

---

## Cost

### Local (free)

- MacBook
- Python
- Silero VAD (via torch)
- mlx-whisper (via huggingface)
- Chatterbox TTS (via mlx-audio)

### Cloud

- Cerebras API (Qwen 3 235B)

No paid realtime communication infrastructure. The phone ↔ Mac audio path is local Wi-Fi.

---

## Future Extensions

The architecture is designed to support:

```
                    Voice Agent
                        │
              ┌─────────┼──────────┐
              │         │          │
            Browser    SIP       Twilio
              │         │          │
            WebRTC    Phone      Phone
```

And eventually:

```
Client
  ↓
Phone number
  ↓
Telephony
  ↓
Voice Agent
  ↓
Tools
  ↓
CRM / Calendar / Database
```

None of this is in V1. The transport interface is designed to make it possible.

---

## License

MIT
