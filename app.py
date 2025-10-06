import os
import json
import random
from typing import Optional

import gradio as gr
from huggingface_hub import InferenceClient

pipe = None           # global local pipeline
tokenizer = None      # global local tokenizer

# ========== Config ==========
# Use a SMALL, instruction-tuned chat model for local mode by default
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Provider: "hf" | "nebius" | (fallback decided by keys below)
API_PROVIDER = os.environ.get("API_PROVIDER", "").strip().lower()

# HF model/task
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", os.environ.get("API_MODEL", "HuggingFaceH4/zephyr-7b-beta")).strip()
HF_TASK = os.environ.get("HF_TASK", "").strip().lower()  # optional override: "conversational" | "text-generation"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Nebius
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
NEBIUS_MODEL = os.environ.get("NEBIUS_MODEL", "openai/gpt-oss-20b")
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
# ===========================

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

    fact = random.choice(WPI_FACTS)["text"]
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"{message}\n\nFun fact: {fact}"} )

    response = ""

    if use_local_model:
        # Local transformers pipeline with chat-aware formatting
        from transformers import pipeline, AutoTokenizer
        import torch

        # Keep CPU from thrashing; use any GPU if present
        try:
            torch.set_num_threads(2)
        except Exception:
            pass

        if pipe is None or tokenizer is None:
            # Lazy init
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, trust_remote_code=True)
            pipe = pipeline(
                "text-generation",
                model=LOCAL_MODEL,
                tokenizer=tokenizer,
                device_map="auto",
                trust_remote_code=True,
            )

        prompt = _build_local_prompt(messages)

        # Generate once (non-stream) and yield the assistant part
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
        # If we used a chat template, slicing by prompt length is safe
        assistant = full[len(prompt):].strip()
        if "Assistant:" in assistant:
            assistant = assistant.split("Assistant:", 1)[-1].strip()
        yield assistant
        return

    provider = _resolve_provider()

    if provider == "nebius":
        print(f"[MODE] api | provider=nebius model={NEBIUS_MODEL}")
        if not NEBIUS_API_KEY:
            yield "⚠️ Missing NEBIUS_API_KEY. Set it or switch to HF by setting API_PROVIDER=hf and providing HF_TOKEN."
            return
        # Use HF client with custom base for Nebius OpenAI-compatible chat
        client = InferenceClient(token=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)
        try:
            for chunk in client.chat_completion(  # type: ignore[attr-defined]
                messages=messages,
                max_tokens=int(max_tokens),
                stream=True,
                temperature=float(temperature),
                top_p=float(top_p),
                model=NEBIUS_MODEL,
            ):
                choices = getattr(chunk, "choices", [])
                token_text = ""
                if choices and getattr(choices[0].delta, "content", None):
                    token_text = choices[0].delta.content
                response += token_text
                yield response
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                yield "⚠️ Nebius auth failed. Check NEBIUS_API_KEY and NEBIUS_MODEL."
            else:
                yield f"⚠️ Nebius API error: {e}"
        return

    # HF provider
    model_id = HF_MODEL_ID
    print(f"[MODE] api | provider=hf model={model_id}")
    token_value = _extract_hf_token(hf_token)
    if not token_value:
        yield "🔐 Please log in to Hugging Face or set HF_TOKEN to use the API path."
        return

    client = InferenceClient(model=model_id, token=token_value)
    task = _hf_task_for_model(model_id)

    if task == "conversational":
        # Non-streaming conversational call (Zephyr etc.)
        try:
            conv = client.conversational(  # type: ignore[attr-defined]
                input="\n".join([f"{m['role']}: {m['content']}" for m in messages]),
                parameters={
                    "max_new_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
            )
            text = getattr(conv, "generated_text", None)
            if text is None and isinstance(conv, dict):
                text = conv.get("generated_text", "")
            yield (text or "").strip()
        except Exception as e:
            if "not supported for task" in str(e).lower():
                yield f"⚠️ HF model '{model_id}' expects task 'conversational'. Set HF_TASK=conversational or choose a text-generation model."
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                yield "⚠️ Hugging Face auth failed. Ensure HF_TOKEN is set or log in via the button."
            else:
                yield f"⚠️ HF Inference error (conversational): {e}"
        return
    else:
        # Streaming text-generation path
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        try:
            stream = client.text_generation(
                prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                stream=True,
                details=False,
                return_full_text=False,
            )
            for out in stream:
                try:
                    token_text = getattr(out, "token", None)
                    token_text = token_text.text if token_text else (out if isinstance(out, str) else "")
                except Exception:
                    token_text = str(out) if out else ""
                response += token_text
                yield response
        except Exception as e:
            if "not supported for task" in str(e).lower():
                yield f"⚠️ HF model '{model_id}' does not support text-generation. Try HF_TASK=conversational (e.g., for Zephyr) or switch HF_MODEL_ID."
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                yield "⚠️ Hugging Face auth failed. Ensure HF_TOKEN or log in via the button."
            else:
                yield f"⚠️ HF Inference error: {e}"

# ---- Build UI only when asked ----
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
                token_input,  # LoginButton or dummy State(None) to keep signature aligned
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
    if "demo" not in globals():
        _enable_oauth = os.getenv("ENABLE_OAUTH", "0").lower() not in ("0", "false", "no")
        demo = create_demo(enable_oauth=_enable_oauth)
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
