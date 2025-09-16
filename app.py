import gradio as gr
from huggingface_hub import InferenceClient
import os
import json
import random
from typing import Optional

pipe = None

# ========== Config ==========
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "microsoft/Phi-3-mini-4k-instruct")

API_PROVIDER = os.environ.get("API_PROVIDER", "").strip().lower()   # "", "hf", "nebius"
API_MODEL = os.environ.get("API_MODEL", "HuggingFaceH4/zephyr-7b-beta")

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
NEBIUS_MODEL = os.environ.get("NEBIUS_MODEL", "gpt-oss-20b")
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

def _resolve_provider():
    if API_PROVIDER in ("hf", "nebius"):
        return API_PROVIDER
    return "nebius" if NEBIUS_API_KEY else "hf"

# ---- Core chat handler (unchanged logic) ----
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
    global pipe

    fact = random.choice(WPI_FACTS)["text"]
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"{message}\n\nFun fact: {fact}"})

    response = ""

    if use_local_model:
        print("[MODE] local")
        from transformers import pipeline
        if pipe is None:
            pipe = pipeline("text-generation", model=LOCAL_MODEL)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()
        return

    provider = _resolve_provider()

    if provider == "nebius":
        print(f"[MODE] api | provider=nebius model={NEBIUS_MODEL}")
        if not NEBIUS_API_KEY:
            yield ("⚠️ Missing NEBIUS_API_KEY. Set it or switch to HF by setting API_PROVIDER=hf and providing HF_TOKEN.")
            return
        client = InferenceClient(token=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)
        try:
            for chunk in client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
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

    # HF provider via text_generation (no strict chat perms)
    print(f"[MODE] api | provider=hf model={API_MODEL}")
    token_value = _extract_hf_token(hf_token)
    if not token_value:
        yield "⚠️ Please log in (Login button) or set HF_TOKEN in environment."
        return
    client = InferenceClient(model=API_MODEL, token=token_value)

    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    try:
        stream = client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
        if "401" in str(e) or "Unauthorized" in str(e):
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
                token_input,  # LoginButton or a dummy State(None) to keep signature aligned
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

# Create demo automatically unless we're in CI/tests
if os.environ.get("SKIP_UI_ON_IMPORT") != "1":
    demo = create_demo(enable_oauth=True)

if __name__ == "__main__":
    # If not created above (e.g., when SKIP_UI_ON_IMPORT=1 locally), create now
    if "demo" not in globals():
        demo = create_demo(enable_oauth=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
