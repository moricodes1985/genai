import gradio as gr
import requests

API_URL = "http://backend:8001/chat"

def chat_fn(history, message, strict):
    if not message.strip():
        return history, ""

    try:
        r = requests.post(API_URL, json={"message": message, "strict": strict}, timeout=30)
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "⚠️ Backend returned no answer.")
    except requests.exceptions.RequestException as e:
        answer = f"⚠️ Request failed: {e}"
    except ValueError:
        answer = "⚠️ Backend returned invalid JSON."

    return (history or []) + [[message, answer]], ""

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# 📚 Open-Source License Copilot (RAG Demo)")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=520)
                msg = gr.Textbox(placeholder="Ask about MIT/Apache/GPL obligations, SPDX metadata, etc.")
                strict = gr.Checkbox(label="Strict mode", value=False)
                send = gr.Button("Send")
            with gr.Column():
                gr.Markdown("### Try queries")
                gr.Markdown("- *What does MIT allow me to do?*")
                gr.Markdown("- *Is GPL-3.0 compatible with MIT?*")

        send.click(chat_fn, [chat, msg, strict], [chat, msg])
        msg.submit(chat_fn, [chat, msg, strict], [chat, msg])  # press Enter to send

    demo.launch(server_name="0.0.0.0", server_port=7860)
