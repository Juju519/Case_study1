import os
import json
import random
import time
from typing import Optional

import gradio as gr
from huggingface_hub import InferenceClient

# --- Prometheus app metrics (port :8000) ---
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info

print("[CS3] STARTUP app.py using requests-based HF Router path")


PRODUCT_KIND = os.getenv("PRODUCT_KIND", "unknown")  # "local" | "api" (set per container)
HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co")

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
    "Approx tokens emitted (len of chunks / 4 heuristic)",
    ["product"],
)
BUILD_INFO = Info(
    "gompei_build_info",
    "Build/provider/model info for this instance",
)

pipe = None           # global local pipeline
tokenizer = None      # global local tokenizer

# ========== Config ==========
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

API_PROVIDER = os.environ.get("API_PROVIDER", "").strip().lower()

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", os.environ.get("API_MODEL", "HuggingFaceH4/zephyr-7b-beta")).strip()
HF_TASK = os.environ.get("HF_TASK", "").strip().lower()
HF_TOKEN = os.environ.get("HF_TOKEN")
# NEW: default to HF router endpoint to avoid 410 Gone on api-inference
HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co/hf-inference")

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
NEBIUS_MODEL = os.environ.get("NEBIUS_MODEL", "openai/gpt-oss-20b")
HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co")
print("[CS3] Router base:", HF_BASE_URL)
print("[CS3] Model id:", os.environ.get("HF_MODEL_ID"))

# ===========================



def _hf_model_url(model_id: str) -> str:
    return f"{HF_BASE_URL.rstrip('/')}/models/{model_id}"

# Facts + CSS fallbacks
FACTS_PATH = "facts.json"
DEFAULT_FACTS = [{"text": "WPI was founded in 1865 by John Boynton and Ichabod Washburn."}]
try:
    with open(FACTS_PATH, "r") as f:
        WPI_FACTS = json.load(f)
    if not isinstance(WPI_FACTS, list) or not WPI_FACTS:
        WPI_FACTS = DEFAULT_FACTS
except Exception:
    WPI_FACTS = DEFAULT_FACTS

fancy_css = """/* fallback if your CSS file isn't ready */ #title { text-align:center; }"""

def _build_hf_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Simple chat-style prompt for HF text_generation."""
    parts = []
    for m in messages:
        role = m["role"]
        if role == "system":
            parts.append(f"System: {m['content']}")
        elif role == "user":
            parts.append(f"User: {m['content']}")
        elif role == "assistant":
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)

def _extract_hf_token(hf_token_obj: Optional[object]) -> Optional[str]:
    """Accepts LoginButton return, dict, or string; falls back to env HF_TOKEN."""
    if hf_token_obj:
        if isinstance(hf_token_obj, str) and hf_token_obj.strip():
            return hf_token_obj.strip()
        for attr in ("token", "access_token"):
            try:
                val = getattr(hf_token_obj, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            except Exception:
                pass
        try:
            if hasattr(hf_token_obj, "get"):
                val = hf_token_obj.get("token") or hf_token_obj.get("access_token")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
    env_val = os.environ.get("HF_TOKEN")
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()
    return None

def _resolve_provider() -> str:
    """Choose provider if not explicitly set."""
    if API_PROVIDER in ("hf", "nebius"):
        return API_PROVIDER
    return "nebius" if NEBIUS_API_KEY else "hf"

def _hf_task_for_model(model_id: str) -> str:
    """Pick the correct HF task: explicit env wins; else detect by model name."""
    if HF_TASK in ("conversational", "text-generation"):
        return HF_TASK
    if "zephyr" in model_id.lower():
        return "conversational"
    return "text-generation"

# -------- Local helpers (instruction-tuned formatting) --------
def _build_local_prompt(msgs: list[dict[str, str]]) -> str:
    """
    Build a chat-style prompt for local models.
    If the tokenizer has a chat template, use it; else fall back to a simple format.
    """
    global tokenizer
    try:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass

    # Fallback generic prompt
    parts = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            parts.append(f"System: {m['content']}")
        elif role == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)

# ---- Core chat handler ----
def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    use_local_model: bool,
    hf_token: Optional[object] = None,
):
    global pipe, tokenizer

    start_time = time.time()
    token_estimate = 0
    status = "ok"

    try:
        fact = random.choice(WPI_FACTS)["text"]
        messages = [{"role": "system", "content": system_message}]
        messages.extend(history)
        messages.append({"role": "user", "content": f"{message}\n\nFun fact: {fact}"})

        response = ""

        if use_local_model:
            # ---- LOCAL MODEL PATH ----
            from transformers import pipeline, AutoTokenizer
            import torch

            try:
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

            prompt = _build_local_prompt(messages)
            outputs = pipe(
                prompt,
                max_new_tokens=int(max_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full = outputs[0]["generated_text"]
            assistant = full[len(prompt):].strip()
            if "Assistant:" in assistant:
                assistant = assistant.split("Assistant:", 1)[-1].strip()

            token_estimate += max(0, len(assistant)) // 4
            yield assistant

        else:
            # ---- HUGGING FACE API PATH via Router (OpenAI-compatible) ----
            model_id = HF_MODEL_ID
            token_value = os.getenv("HF_TOKEN", "").strip()
            if not token_value:
                status = "error"
                yield "🔐 Please log in to Hugging Face or set HF_TOKEN to use the API path."
            else:
                prompt = _build_hf_chat_prompt(messages)

                url = f"{HF_BASE_URL.rstrip('/')}/v1/completions"
                headers = {"Authorization": f"Bearer {token_value}"}
                payload = {
                    "model": model_id,
                    "prompt": prompt,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p)
                }
                try:
                    t0 = time.time()
                    import requests
                    r = requests.post(url, headers=headers, json=payload, timeout=120)
                    if r.status_code == 401:
                        status = "error"
                        yield "⚠️ Hugging Face auth failed (401). Ensure HF_TOKEN is valid and model terms are accepted."
                    elif r.status_code == 404:
                        status = "error"
                        yield ("⚠️ Model not found at the router (404). "
                               "Double-check HF_MODEL_ID or try `mistralai/Mistral-7B-Instruct-v0.3`.")
                    elif r.status_code >= 400:
                        status = "error"
                        yield f"⚠️ HF Router error {r.status_code}: {r.text[:300]}"
                    else:
                        data = r.json()
                        text = ""
                        try:
                            text = data["choices"][0].get("text") or ""
                        except Exception:
                            text = str(data)[:500]
                        RESP_LATENCY.observe(time.time() - t0)
                        TOKENS_OUT.labels(PRODUCT_KIND).inc(max(0, len(text)) // 4)
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
        # ---- METRICS UPDATE ----
        REQS_TOTAL.labels(PRODUCT_KIND, status).inc()
        ACTIVE_SESSIONS.set(0 if not history else len(history))
        RESP_LATENCY.observe(time.time() - start_time)
        TOKENS_OUT.labels(PRODUCT_KIND).inc(token_estimate)


def create_demo(enable_oauth: bool = True):
    with gr.Blocks(css=fancy_css) as demo:
        with gr.Row():
            gr.Markdown("<h1 id='title'>🐐 Chat with Gompei</h1>")
            token_input = gr.LoginButton() if enable_oauth else gr.State(value=None)

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
if os.environ.get("SKIP_UI_ON_IMPORT") != "1":
    _enable_oauth = os.getenv("ENABLE_OAUTH", "0").lower() not in ("0", "false", "no")
    demo = create_demo(enable_oauth=_enable_oauth)

if __name__ == "__main__":
    # Start Prometheus metrics server on :8000 before launching UI
    start_http_server(8000)
    BUILD_INFO.info({
        "version": "cs3",
        "provider": os.getenv("API_PROVIDER", "local"),
        "local_model": os.getenv("LOCAL_MODEL", ""),
        "hf_model_id": os.getenv("HF_MODEL_ID", ""),
        "nebius_model": os.getenv("NEBIUS_MODEL", ""),
        "product": PRODUCT_KIND,
    })

    if "demo" not in globals():
        _enable_oauth = os.getenv("ENABLE_OAUTH", "0").lower() not in ("0", "false", "no")
        demo = create_demo(enable_oauth=_enable_oauth)

    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
