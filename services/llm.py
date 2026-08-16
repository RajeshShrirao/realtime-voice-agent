import re
import time

import config
import state
import utils


def _strip_thinking_tokens(text: str) -> tuple[str, str]:
    thought = ""
    m = re.search(r'<\|channel>(.*?)(?:<channel\|>|$)', text, re.DOTALL)
    if m:
        thought = m.group(1).strip()
    clean = re.sub(r'<\|channel>.*?(?:<channel\|>|$)', '', text, flags=re.DOTALL)
    return clean.strip(), thought


def generate_text(messages, cerebras_client, max_tokens=512):
    if cerebras_client is None:
        return ""

    attempt = 0
    while attempt < config.LLM_MAX_RETRIES:
        try:
            resp = cerebras_client.chat.completions.create(
                model=config.LLM_MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
            )
            raw = resp.choices[0].message.content or ""
            clean, _ = _strip_thinking_tokens(raw)
            return clean.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str or "queue_exceeded" in err_str:
                if attempt < config.LLM_MAX_RETRIES - 1:
                    delay = config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"\n  [retry {attempt+1}/{config.LLM_MAX_RETRIES}] rate limited, waiting {delay}s...", end="", flush=True)
                    time.sleep(delay)
                    print(" retrying...", end=" ", flush=True)
                    attempt += 1
                    continue
            utils.record_stat("errors", f"LLM API: {e}")
            return ""
        attempt += 1
    return ""


def generate_stream(messages, cerebras_client, tts_queue):
    if cerebras_client is None:
        print("  [ERROR] No Cerebras client initialized", flush=True)
        return ""

    full = ""
    clean_text = ""
    buffer = ""
    first_token = False
    llm_stream_start = time.perf_counter()

    kwargs = {
        "model": config.LLM_MODEL_ID,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "top_p": config.LLM_TOP_P,
        "stream": True,
        "max_completion_tokens": 512,
    }

    attempt = 0
    while attempt < config.LLM_MAX_RETRIES:
        try:
            with utils.measure_time("llm_total_ms"):
                stream = cerebras_client.chat.completions.create(**kwargs)

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta.content:
                        continue

                    token = delta.content
                    if not first_token:
                        first_token = True
                        utils.record_stat("llm_first_token_ms", (time.perf_counter() - llm_stream_start) * 1000)

                    full += token
                    utils.record_stat("llm_tokens", increment=1)

                    new_clean, _ = _strip_thinking_tokens(full)
                    new_text = new_clean[len(clean_text):]
                    clean_text = new_clean

                    if config.TEXT_MODE and new_text:
                        print(new_text, end="", flush=True)

                    buffer += new_text

                    clause_triggers = (".", "!", "?", "...", "\n", "—")
                    if any(buffer.rstrip().endswith(p) for p in clause_triggers):
                        if len(buffer.strip()) > 20:
                            tts_queue.put(buffer.strip())
                            buffer = ""

                break

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str or "queue_exceeded" in err_str:
                if attempt < config.LLM_MAX_RETRIES - 1:
                    delay = config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"\n  [retry {attempt+1}/{config.LLM_MAX_RETRIES}] rate limited, waiting {delay}s...", end="", flush=True)
                    time.sleep(delay)
                    print(" retrying...", end=" ", flush=True)
                    attempt += 1
                    continue
            utils.record_stat("errors", f"LLM stream: {e}")
            print(f"\n  [ERROR] Stream failed: {e}", flush=True)
            return ""

    if buffer.strip():
        tts_queue.put(buffer.strip())

    final_clean, _ = _strip_thinking_tokens(full)
    if not config.TEXT_MODE:
        print(final_clean.strip(), flush=True)

    return final_clean
