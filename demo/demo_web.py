import gradio as gr
import base64
import time
import random
from openai import OpenAI

# --------------------
# 配置
# --------------------
API_KEY = "sk-VyrJiYLZRV1Vo6eNJ1V4EvGDa1tAVKpF"  # 你的 key
API_BASE = "http://172.18.65.239:8000/v1"        # 你的本地部署地址
MODEL_NAME = "o4-mini-2025-04-16"                # 模型名字

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 3


# --------------------
# 辅助函数
# --------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def gpt_api_call(image_path, user_prompt):
    """调用你自己的本地 API 生成 HTML"""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    img_b64 = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": user_prompt.strip()}
            ]
        }
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
                time.sleep(delay)
            else:
                return f"❌ Error after {MAX_RETRIES} attempts: {e}"


# --------------------
# Gradio 前端逻辑
# --------------------
def ui_to_html(image, prompt_text):
    if image is None:
        return "Please upload a UI image first.", ""
    if not prompt_text.strip():
        prompt_text = "Generate the corresponding HTML code for this UI image."

    image.save("temp_input.png")
    result = gpt_api_call("temp_input.png", prompt_text)
    preview_html = f"<iframe style='width:100%;height:500px;border:1px solid #ccc' srcdoc=\"{result}\"></iframe>"
    return result, preview_html


# --------------------
# 启动 Gradio App
# --------------------
demo = gr.Interface(
    fn=ui_to_html,
    inputs=[
        gr.Image(label="Upload UI Screenshot", type="pil"),
        gr.Textbox(
            label="🧠 Custom Prompt (你可以自己写指令)",
            value="Generate the corresponding HTML code for this UI image.",
            lines=3,
            placeholder="例如：Please generate TailwindCSS code for this UI layout."
        )
    ],
    outputs=[
        gr.Code(label="Generated HTML Code", language="html"),
        gr.HTML(label="Rendered Preview")
    ],
    title="🎨 UI2Code Local Demo",
    description="上传一张UI截图，自定义Prompt生成HTML，并实时预览。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
