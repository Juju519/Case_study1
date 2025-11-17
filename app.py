# app.py (CS3 compliant, with safe fallbacks)
# - Prometheus metrics on :8000
# - Gradio UI on :7860 (0.0.0.0 for Docker)
# - Local vs API selection via checkbox + env
# - Falls back to local model if no API creds
# - Avoids double-counting RESP_LATENCY

import os
import json
import random
import time
from typing import Optional

import gradio as gr
import requests

from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info

print("[CS3] STARTUP")

# ========== Config ==========

PRODUCT_KIND = os.getenv("PRODUCT_KIND", "unknown")  # "local" | "api" | "unknown"

# Local model
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "sshleifer/tiny-gpt2").strip()

# OpenAI-compatible provider (OpenRouter / Together / OpenAI)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Hugging Face Router fallback
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co").strip()
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/gemma-2-2b-it").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

print(f"[CS3] PRODUCT_KIND={PRODUCT_KIND}")
print(f"[CS3] LOCAL_MODEL={LOCAL_MODEL}")
print(f"[CS3] OPENAI_BASE_URL={'<set>' if OPENAI_BASE_URL else '<empty>'}")
print(f"[CS3] HF_BASE_URL={HF_BASE_URL}")
print(f"[CS3] HF_MODEL_ID={HF_MODEL_ID}")
print(f"[CS3] HF_TOKEN={'<set>' if HF_TOKEN else '<empty>'}")

# ========== Metrics ==========

REQS_TOTAL = Counter(
    "gompei_requests_total",
    "Total chat requests processed",
    ["product", "status"],
)
RESP_LATENCY = Histogram(
    "gompei_response_latency_seconds",
    "End-to-end response latency (seconds)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)
ACTIVE_SESSIONS = Gauge(
    "gompei_active_sessions",
    "Active chat sessions (len(history) proxy)",
)
TOKENS_OUT = Counter(
    "gompei_tokens_emitted_total",
    "Approx tokens emitted (chars/4 heuristic)",
    ["product"],
)
BUILD_INFO = Info(
    "gompei_build_info",
    "Build/provider/model info for this instance",
)

# ========== Facts + CSS ==========

FACTS_PATH = "facts.json"
DEFAULT_FACTS = [{"text": "WPI was founded in 1865 by John Boynton and Ichabod Washburn."}]

try:
    with open(FACTS_PATH, "r") as f:
        WPI_FACTS = json.load(f)
    if not isinstance(WPI_FACTS, list) or not WPI_FACTS:
        WPI_FACTS = DEFAULT_FACTS
except Exception as e:
    print(f"[CS3] Could not load facts.json: {e}")
    WPI_FACTS = DEFAULT_FACTS

fancy_css = "#title { text-align: center; }"

# ========== Prompt helpers ==========


def _build_local_prompt(msgs: list[dict[str, str]]) -> str:
    """Simple chat-ish prompt for local text-generation."""
    parts = []
    for m in msgs:
        r = m["role"]
        if r == "system":
            parts.append(f"System: {m['content']}")
        elif r == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _build_chat_messages(
    system_message: str,
    history: list[dict[str, str]],
    user_text: str,
) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system_message}]
    msgs.extend(history or [])
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _build_hf_chat_prompt(msgs: list[dict[str, str]]) -> str:
    """Flat prompt for HF /v1/completions."""
    parts = []
    for m in msgs:
        r = m["role"]
        if r == "system":
            parts.append(f"System: {m['content']}")
        elif r == "user":
            parts.append(f"User: {m['content']}")
        elif r == "assistant":
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ========== State for local model ==========

pipe = None
tokenizer = None


# ========== Core chat handler ==========


def respond(
    message: str,
    history: list[dict[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    use_local_model: bool,
    _unused_login: Optional[object] = None,  # kept for signature compatibility
):
    """
    Main chat handler:
    - If use_local_model is True → always local HF transformers pipeline.
    - Else:
        - If OPENAI_API_KEY set → OpenAI-compatible API path.
        - Elif HF_TOKEN set → Hugging Face Router /v1/completions.
        - Else → fall back to local model (no more 🔐 error).
    """
    global pipe, tokenizer

    start_time = time.time()
    token_estimate = 0
    status = "ok"

    try:
        # Pick a random WPI fact
        fact = random.choice(WPI_FACTS)["text"]
        user_with_fact = f"{message}\n\nFun fact: {fact}"

        # Effective decision: if no API creds at all, force local
        effective_use_local = use_local_model
        if not effective_use_local and not OPENAI_API_KEY and not HF_TOKEN:
            print("[CS3] No OPENAI_API_KEY or HF_TOKEN found; falling back to local model.")
            effective_use_local = True

        # ---------- LOCAL MODEL PATH ----------
        if effective_use_local:
            print("[CS3] MODE=local")
            try:
                from transformers import pipeline, AutoTokenizer
                try:
                    import torch
                    torch.set_num_threads(2)
                except Exception:
                    pass

                if pipe is None or tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        LOCAL_MODEL,
                        trust_remote_code=True,
                    )
                    pipe = pipeline(
                        "text-generation",
                        model=LOCAL_MODEL,
                        tokenizer=tokenizer,
                        device_map="auto",
                        trust_remote_code=True,
                    )

                local_msgs = (
                    [{"role": "system", "content": system_message}]
                    + (history or [])
                    + [{"role": "user", "content": user_with_fact}]
                )
                prompt = _build_local_prompt(local_msgs)
                outputs = pipe(
                    prompt,
                    max_new_tokens=int(max_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
                full = outputs[0]["generated_text"]
                assistant = full[len(prompt):].strip()
                if "Assistant:" in assistant:
                    assistant = assistant.split("Assistant:", 1)[-1].strip()

                token_estimate += max(0, len(assistant)) // 4
                yield assistant
                return

            except Exception as e:
                status = "error"
                print(f"[CS3] Local model failed: {e}")
                yield f"⚠️ Local model error: {e}"
                return

        # ---------- API PATH (OPENAI-COMPATIBLE) ----------
        if OPENAI_API_KEY:
            print("[CS3] MODE=api (OpenAI-compatible)")
            base = (OPENAI_BASE_URL or "https://openrouter.ai/api").rstrip("/")
            url = f"{base}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                # optional headers for OpenRouter, safe to send elsewhere too
                "HTTP-Referer": os.getenv("OR_REFERER", "http://localhost"),
                "X-Title": os.getenv("OR_TITLE", "Gompei CS3"),
            }
            msgs = _build_chat_messages(system_message, history or [], user_with_fact)
            payload = {
                "model": HF_MODEL_ID,  # reuse env var to choose model name
                "messages": msgs,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }

            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 401:
                    status = "error"
                    yield "⚠️ Auth failed (401). Check OPENAI_API_KEY and model access."
                elif r.status_code >= 400:
                    status = "error"
                    yield f"⚠️ API error {r.status_code}: {r.text[:300]}"
                else:
                    data = r.json()
                    text = data["choices"][0]["message"]["content"]
                    token_estimate += max(0, len(text)) // 4
                    yield text
            except requests.Timeout:
                status = "error"
                yield "⚠️ API timeout. Try again or lower max tokens."
            except Exception as e:
                status = "error"
                yield f"⚠️ API request failed: {e}"
            return

        # ---------- API PATH (HF ROUTER) ----------
        # We know OPENAI_API_KEY is empty here. Use HF_TOKEN if available.
        if HF_TOKEN:
            print("[CS3] MODE=api (HF Router)")
            url = f"{HF_BASE_URL.rstrip('/')}/v1/completions"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            msgs = _build_chat_messages(system_message, history or [], user_with_fact)
            prompt = _build_hf_chat_prompt(msgs)
            payload = {
                "model": HF_MODEL_ID,
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }

            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 401:
                    status = "error"
                    yield "⚠️ Hugging Face auth failed (401). Ensure HF_TOKEN is valid."
                elif r.status_code == 404:
                    status = "error"
                    yield (
                        "⚠️ Model not found at HF Router (404). "
                        "Try a widely available model like "
                        "`google/gemma-2-2b-it` or adjust HF_MODEL_ID."
                    )
                elif r.status_code >= 400:
                    status = "error"
                    yield f"⚠️ HF Router error {r.status_code}: {r.text[:300]}"
                else:
                    data = r.json()
                    text = data["choices"][0].get("text") or ""
                    token_estimate += max(0, len(text)) // 4
                    yield text
            except requests.Timeout:
                status = "error"
                yield "⚠️ HF Router timeout. Try again or lower max tokens."
            except Exception as e:
                status = "error"
                yield f"⚠️ HF request failed: {e}"
            return

        # If we somehow get here, fall back to local with a message
        status = "error"
        yield "⚠️ No API credentials found; please enable 'Use Local Model' or set OPENAI_API_KEY/HF_TOKEN."

    except Exception:
        status = "error"
        raise
    finally:
        # One end-to-end latency observation per request
        RESP_LATENCY.observe(time.time() - start_time)
        REQS_TOTAL.labels(PRODUCT_KIND, status).inc()
        ACTIVE_SESSIONS.set(0 if not history else len(history))
        TOKENS_OUT.labels(PRODUCT_KIND).inc(token_estimate)


# ========== Gradio UI ==========


def create_demo(enable_oauth: bool = False):
    with gr.Blocks(css=fancy_css) as demo:
        with gr.Row():
            gr.Markdown("<h1 id='title'>🐐 Chat with Gompei</h1>")
            # CS3: OAuth disabled; keep a dummy state to match fn signature
            token_input = gr.State(value=None)

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                gr.Textbox(
                    value=(
                        "You are Gompei the Goat, WPI's mascot. "
                        "Answer questions with fun goat-like personality and real WPI facts."
                    ),
                    label="System message",
                ),
                gr.Slider(
                    minimum=1,
                    maximum=1024,
                    value=256,
                    step=1,
                    label="Max new tokens",
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p (nucleus sampling)",
                ),
                gr.Checkbox(label="Use Local Model", value=False),
                token_input,  # placeholder for _unused_login
            ],
            type="messages",
            examples=[
                [
                    "Where is WPI located?",
                    (
                        "You are Gompei the Goat, WPI's mascot. "
                        "Answer questions with fun goat-like personality and real WPI facts."
                    ),
                    128,
                    0.7,
                    0.95,
                    False,
                    None,
                ],
                [
                    "Who founded WPI?",
                    (
                        "You are Gompei the Goat, WPI's mascot. "
                        "Answer questions with fun goat-like personality and real WPI facts."
                    ),
                    128,
                    0.7,
                    0.95,
                    False,
                    None,
                ],
            ],
        )
    return demo


# Auto-create UI unless tests/CI skip it
if os.getenv("SKIP_UI_ON_IMPORT") != "1":
    demo = create_demo(enable_oauth=False)

if __name__ == "__main__":
    # Start Prometheus metrics server on :8000 before launching UI
    start_http_server(8000)
    BUILD_INFO.info({
        "version": "cs3",
        "provider": (
            "openai-compatible" if OPENAI_API_KEY
            else ("hf-router" if HF_TOKEN else "local-only")
        ),
        "local_model": LOCAL_MODEL,
        "hf_model_id": HF_MODEL_ID,
        "product": PRODUCT_KIND,
    })
    if "demo" not in globals():
        demo = create_demo(enable_oauth=False)
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
