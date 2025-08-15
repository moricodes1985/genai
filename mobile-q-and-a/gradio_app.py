import os, uuid, json, httpx, gradio as gr
from dotenv import load_dotenv

load_dotenv()
print("Gradio version:", gr.__version__)

API_BASE = os.getenv("API_BASE", "http://api:8000")
ASK_URL = f"{API_BASE.rstrip('/')}/ask"

def ask_backend_sync(user_id: str, question: str):
    r = httpx.post(
        ASK_URL,
        headers={"Content-Type": "application/json"},
        content=json.dumps({"user_id": user_id, "question": question}),
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()

def on_apply_base(new_base: str):
    global ASK_URL
    ASK_URL = f"{new_base.rstrip('/')}/ask"
    return gr.update(value=new_base), gr.update(value=f"Calling: {ASK_URL}")

def ensure_tuples(history):
    """Coerce history into list[tuple[str,str]] for Chatbot(type='tuples')."""
    if not history:
        return []
    # history might come back as list[list[str,str]]; make them tuples
    if isinstance(history, list) and history and isinstance(history[0], (list, tuple)):
        return [ (str(u), str(b)) for (u, b) in history ]
    # or (rarely) messages dicts -> convert to tuples
    if isinstance(history, list) and history and isinstance(history[0], dict):
        pairs = []
        buf_user = None
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                buf_user = content
            elif role == "assistant":
                pairs.append((buf_user or "", content))
                buf_user = None
        return pairs
    return []

def on_send(message, history, user_state):
    history = ensure_tuples(history)
    if not user_state or "user_id" not in user_state:
        user_state = {"user_id": f"u-{uuid.uuid4().hex[:8]}"}

    msg = (message or "").strip()
    if not msg:
        return history, user_state, gr.update()

    try:
        data = ask_backend_sync(user_state["user_id"], msg)
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        if sources:
            src_lines = [
                f"{i}. **{s.get('source','policy.txt')}** — {s.get('snippet','')}"
                for i, s in enumerate(sources, 1)
            ]
            answer += "\n\n---\n**Sources**\n" + "\n".join(src_lines)

        history = history + [(msg, answer)]
        return history, user_state, gr.update(value="")  # clear input
    except Exception as e:
        history = history + [(msg, f"⚠️ {type(e).__name__}: {e}")]
        return history, user_state, gr.update()

def on_clear():
    return []

with gr.Blocks(title="RAG Q&A (FAISS)") as demo:
    gr.Markdown("# 🧠 RAG Q&A Chatbot\nAsk questions about the company policy.\nThis UI calls the FastAPI backend at `/ask`.")

    with gr.Row():
        api_base_box = gr.Textbox(label="API Base", value=API_BASE, interactive=True, scale=5)
        apply_btn = gr.Button("Apply", scale=1)
    call_info = gr.Markdown(value=f"Calling: {ASK_URL}")

    user_state = gr.State({"user_id": f"u-{uuid.uuid4().hex[:8]}"})

    # ✅ Use tuples mode (plural) and return list[(user, assistant)]
    chat = gr.Chatbot(height=420, show_copy_button=True, type="tuples")
    msg = gr.Textbox(placeholder="Type your question…", lines=2, autofocus=True)
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    apply_btn.click(on_apply_base, inputs=[api_base_box], outputs=[api_base_box, call_info])
    send_btn.click(on_send, inputs=[msg, chat, user_state], outputs=[chat, user_state, msg])
    msg.submit(on_send, inputs=[msg, chat, user_state], outputs=[chat, user_state, msg])
    clear_btn.click(on_clear, inputs=None, outputs=[chat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=False)
