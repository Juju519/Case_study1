# app.py (CS3 compliant)
# - Exposes Python Prometheus metrics on :8000
# - Gradio UI on :7860 (binds 0.0.0.0 for Docker)
# - Distinguishes local vs API product via PRODUCT_KIND env
# - Avoids double-counting RESP_LATENCY (observed once in finally)

import os, json, random, time
from typing import Optional

import gradio as gr
import requests

# --- Prometheus app metrics (port :8000) ---
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info

print("[CS3] STARTUP (requests-based API path)")

# ========== Config ==========
PRODUCT_KIND   = os.getenv("PRODUCT_KIND", "unknown")  # "local" | "api" (set per container)

# Local model (only used when the UI checkbox "Use Local Model" is on)
LOCAL_MODEL    = os.getenv("LOCAL_MODEL", "sshleifer/tiny-gpt2").strip()

# OpenAI-compatible provider (preferred if key is present)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()  # e.g., https://openrouter.ai/api or https://api.together.xyz
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()

# Hugging Face Router fallback (if no OpenAI key)
HF_BASE_URL    = os.getenv("HF_BASE_URL", "https://router.huggingface.co").strip()
HF_MODEL_ID    = os.getenv("HF_MODEL_ID", "google/gemma-2-2b-it").strip()
HF_TOKEN       = os.getenv("HF_TOKEN", "").strip()

print(f"[CS3] PRODUCT_KIND={PRODUCT_KIND}")
print(f"[CS3] OPENAI_BASE_URL={'<set>' if OPENAI_BASE_URL else '<empty>'}")
print(f"[CS3] HF_BASE_URL={HF_BASE_URL}")
print(f"[CS3] HF_MODEL_ID={HF_MODEL_ID}")

# ========== Metrics ==========
REQS_TOTAL = Counter("gompei_requests_total", "Total chat requests processed", ["product", "status"])
RESP_LATENCY = Histogram(
    "gompei_response_latency_seconds",
    "End-to-end response latency (seconds)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)
ACTIVE_SESSIONS = Gauge("gompei_active_sessions", "Active chat sessions (len(history) proxy)")
TOKENS_OUT = Counter("gompei_tokens_emitted_total", "Approx tokens emitted (chars/4 heuristic)", ["product"])
BUILD_INFO = Info("gompei_build_info", "Build/provider/model info for this instance")

# ========== Facts + CSS ==========
FACTS_PATH = "facts.json"
DEFAULT_FACTS = [{"text": "WPI was founded in 1865 by John Boynton and Ichabod Washburn."}]
try:
    with open(FACTS_PATH, "r") as f:
        WPI_FACTS = json.load(f)
    if not isinstance(WPI_FACTS, list) or not WPI_FACTS:
        WPI_FACTS = DEFAULT_FACTS
except Exception:
    WPI_FACTS = DEFAULT_FACTS

fancy_css = "#title { text-align:center; }"

# -------- Helpers --------

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


def _build_chat_messages(system_message: str, history: list[dict[str, str]], user_text: str):
    """Build OpenAI-style chat messages."""
    msgs = [{"role": "system", "content": system_message}]
    msgs.extend(history or [])
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _build_hf_chat_prompt(msgs: list[dict[str, str]]) -> str:
    """Simple chat-style prompt for HF /v1/completions."""
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

# ---- Core chat handler ----
pipe = None
tokenizer = None


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    use_local_model: bool,
    _unused_login: Optional[object] = None,   # OAuth disabled; keep arg to satisfy Gradio signature
):
    global pipe, tokenizer

    start_time = time.time()
    token_estimate = 0
    status = "ok"

    try:
        fact = random.choice(WPI_FACTS)["text"]
        user_with_fact = f"{message}\n\nFun fact: {fact}"

        # ---- LOCAL MODEL PATH ----
        if use_local_model:
            from transformers import pipeline, AutoTokenizer
            try:
                import torch
                torch.set_num_threads(2)
            except Exception:
                pass

            if pipe is None or tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, trust_remote_code=True)
                pipe = pipeline(
                    "text-generation",
                    model=LOCAL_MODEL,
                    tokenizer=tokenizer,
                    device_map="auto",
                    trust_remote_code=True,
                )

            local_msgs = [{"role": "system", "content": system_message}] + (history or []) + [
                {"role": "user", "content": user_with_fact}
            ]
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

        # ---- API PATH (OpenAI-compatible preferred; else HF Router) ----
        # Prefer OpenAI-compatible if an API key is available
        if OPENAI_API_KEY:
            base = (OPENAI_BASE_URL or "https://openrouter.ai/api").rstrip("/")
            url = f"{base}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "HTTP-Referer": os.getenv("OR_REFERER", "http://localhost"),
                "X-Title": os.getenv("OR_TITLE", "Gompei CS3"),
            }
            msgs = _build_chat_messages(system_message, history or [], user_with_fact)
            payload = {
                "model": HF_MODEL_ID,  # reuse same env var to pick model name
                "messages": msgs,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 401:
                    status = "error"
                    yield "⚠️ Auth failed (401). Check OPENAI_API_KEY (OpenRouter/Together/OpenAI) and model access."
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

        # Else: Hugging Face Router (requires HF_TOKEN)
        if not HF_TOKEN:
            status = "error"
            yield "🔐 Please log in to Hugging Face or set HF_TOKEN to use the API path."
            return

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
                yield "⚠️ Hugging Face auth failed (401). Ensure HF_TOKEN is valid and model terms are accepted."
            elif r.status_code == 404:
                status = "error"
                yield ("⚠️ Model not found at router (404). "
                       "Try a public model like `google/gemma-2-2b-it`, `mistralai/Mistral-7B-Instruct-v0.3`, "
                       "or switch to an OpenAI-compatible provider via OPENAI_API_KEY.")
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

    except Exception:
        status = "error"
        raise
    finally:
        # One end-to-end latency observation per request
        RESP_LATENCY.observe(time.time() - start_time)
        REQS_TOTAL.labels(PRODUCT_KIND, status).inc()
        ACTIVE_SESSIONS.set(0 if not history else len(history))
        TOKENS_OUT.labels(PRODUCT_KIND).inc(token_estimate)


def create_demo(enable_oauth: bool = False):
    with gr.Blocks(css=fancy_css) as demo:
        with gr.Row():
            gr.Markdown("<h1 id='title'>🐐 Chat with Gompei</h1>")
            token_input = gr.State(value=None)  # OAuth disabled; keep state slot

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                gr.Textbox(
                    value="You are Gompei the Goat, WPI's mascot. Answer questions with fun goat-like personality and real WPI facts.",
                    label="System message",
                ),
                gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="Max new tokens"),
                gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
                gr.Checkbox(label="Use Local Model", value=False),
                token_input,
            ],
            type="messages",
            examples=[
                [
                    "Where is WPI located?",
                    "You are Gompei the Goat, WPI's mascot. Answer questions with fun goat-like personality and real WPI facts.",
                    128, 0.7, 0.95, False, None
                ],
                [
                    "Who founded WPI?",
                    "You are Gompei the Goat, WPI's mascot. Answer questions with fun goat-like personality and real WPI facts.",
                    128, 0.7, 0.95, False, None
                ],
            ],
        )
    return demo

# Auto-create UI unless tests/CI ask us not to
if os.getenv("SKIP_UI_ON_IMPORT") != "1":
    demo = create_demo(enable_oauth=False)

if __name__ == "__main__":
    # Start Prometheus metrics server on :8000 before launching UI
    start_http_server(8000)
    BUILD_INFO.info({
        "version": "cs3",
        "provider": "openai-compatible" if OPENAI_API_KEY else "hf-router",
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
