# import gradio as gr
# import base64
# import time
# import random
# import os
# import webbrowser
# from openai import OpenAI

# # --------------------
# # 配置
# # --------------------
# API_KEY = "sk-xxxx"  # 建议改用环境变量 os.getenv("OPENAI_API_KEY")
# API_BASE = "http://172.18.65.239:8000/v1"
# MODEL_NAME = "o4-mini-2025-04-16"

# MAX_RETRIES = 3
# INITIAL_RETRY_DELAY = 1
# MAX_RETRY_DELAY = 3


# # --------------------
# # 辅助函数
# # --------------------
# def encode_image(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")


# def gpt_api_call(image_path, user_prompt):
#     """调用本地模型 API"""
#     client = OpenAI(api_key=API_KEY, base_url=API_BASE)
#     img_b64 = encode_image(image_path)

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
#                 {"type": "text", "text": user_prompt.strip()}
#             ]
#         }
#     ]

#     for attempt in range(MAX_RETRIES):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=messages,
#                 max_tokens=8192
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"⚠️ Attempt {attempt+1} failed: {e}")
#             if attempt < MAX_RETRIES - 1:
#                 delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
#                 time.sleep(delay)
#             else:
#                 return f"❌ Error after {MAX_RETRIES} attempts: {e}"


# # --------------------
# # 核心逻辑
# # --------------------
# def ui_to_html(image, prompt_text):
#     if image is None:
#         return "⚠️ Please upload a UI image first.", None, None
#     if not prompt_text.strip():
#         prompt_text = "Generate the corresponding HTML code for this UI image."

#     image.save("temp_input.png")
#     result = gpt_api_call("temp_input.png", prompt_text)

#     # 提取 HTML 代码部分
#     html_code = result
#     if "```html" in result:
#         html_code = result.split("```html")[-1].split("```")[0].strip()
#     elif "```" in result:
#         html_code = result.split("```")[1].strip()

#     # 保存 HTML 文件
#     output_path = "generated_ui.html"
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html_code)

#     # 构建 HTML 预览
#     preview_html = f"""
#     <iframe
#         style="width:100%;height:650px;border:none;box-shadow:0 0 10px rgba(0,0,0,0.15);border-radius:12px;"
#         srcdoc="{html_code.replace('"', '&quot;').replace("'", '&apos;')}"
#     ></iframe>
#     """

#     # 返回模型输出全文（上） + 预览（下） + 下载文件
#     return result, preview_html, output_path


# def open_in_browser(file_path):
#     """打开生成的 HTML 文件"""
#     abs_path = os.path.abspath(file_path)
#     webbrowser.open(f"file://{abs_path}")
#     return f"✅ Opened in browser: {abs_path}"


# # --------------------
# # Gradio 界面
# # --------------------
# with gr.Blocks(title="🎨 UI2Code Local Demo") as demo:
#     gr.Markdown("## 🎨 UI2Code Local Demo")
#     gr.Markdown("Upload a UI screenshot, generate structured HTML, preview it live, or open it in your browser.")

#     with gr.Row():
#         # 左边输入
#         with gr.Column(scale=1):
#             image_input = gr.Image(label="Upload UI Screenshot", type="pil")
#             prompt_input = gr.Textbox(
#                 label="🧠 Custom Prompt",
#                 value="Generate the corresponding HTML code for this UI image.",
#                 lines=3,
#                 placeholder="e.g. Generate TailwindCSS code for this layout."
#             )
#             run_btn = gr.Button("🚀 Generate HTML")

#         # 右边输出
#         with gr.Column(scale=1):
#             # 上：模型原始输出（完整内容）
#             model_output = gr.Textbox(
#                 label="🧾 Model Output (Full Response)",
#                 placeholder="The raw text output from the model will appear here...",
#                 lines=12,
#                 interactive=False
#             )
#             # 下：渲染 HTML
#             preview_output = gr.HTML(label="💻 Rendered HTML Preview")

#             with gr.Row():
#                 download_output = gr.File(label="⬇️ Download HTML File")
#                 open_btn = gr.Button("🌐 Preview in Browser")

#             open_status = gr.Textbox(label="Status", interactive=False)

#     # 点击“生成”按钮时触发
#     run_btn.click(
#         fn=ui_to_html,
#         inputs=[image_input, prompt_input],
#         outputs=[model_output, preview_output, download_output]
#     )

#     # 点击“浏览器预览”按钮时触发
#     open_btn.click(
#         fn=open_in_browser,
#         inputs=[download_output],
#         outputs=[open_status]
#     )

# demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


import gradio as gr
import base64
import time
import random
import os
import webbrowser
from openai import OpenAI

# --------------------
# 配置
# --------------------
API_KEY = "sk-xxxx"  # ⚠️ 建议改用环境变量 os.getenv("OPENAI_API_KEY")
API_BASE = "http://172.18.65.239:8000/v1"
MODEL_NAME = "o4-mini-2025-04-16"

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
    """调用本地模型 API"""
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
# 核心逻辑
# --------------------
def ui_to_html(image, prompt_text):
    if image is None:
        return "⚠️ Please upload a UI image first.", None
    if not prompt_text.strip():
        prompt_text = "Generate the corresponding HTML code for this UI image."

    image.save("temp_input.png")
    result = gpt_api_call("temp_input.png", prompt_text)

    # 提取 HTML 部分
    html_code = result
    if "```html" in result:
        html_code = result.split("```html")[-1].split("```")[0].strip()
    elif "```" in result:
        html_code = result.split("```")[1].strip()

    # 保存 HTML 文件
    output_path = "generated_ui.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_code)

    return result, output_path


def open_in_browser(file_path):
    """打开生成的 HTML 文件"""
    if not file_path:
        return "⚠️ No file to open."
    abs_path = os.path.abspath(file_path)
    webbrowser.open(f"file://{abs_path}")
    return f"✅ Opened in browser: {abs_path}"


# --------------------
# Gradio 界面
# --------------------
with gr.Blocks(title="🎨 UI2Code Local Demo") as demo:
    gr.Markdown("## 🎨 UI2Code Local Demo")
    gr.Markdown("Upload a UI screenshot, generate HTML code, download it, or preview it in your browser.")

    with gr.Row():
        # 左边：输入区
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload UI Screenshot", type="pil")
            prompt_input = gr.Textbox(
                label="🧠 Custom Prompt",
                value="Generate the corresponding HTML code for this UI image.",
                lines=3,
                placeholder="e.g. Generate TailwindCSS code for this layout."
            )
            run_btn = gr.Button("🚀 Generate HTML")

        # 右边：输出区
        with gr.Column(scale=1):
            model_output = gr.Textbox(
                label="🧾 Model Output (Full Text)",
                placeholder="Model output (including explanations and HTML code) will appear here...",
                lines=18,
                interactive=False
            )

            with gr.Row():
                download_output = gr.File(label="⬇️ Download HTML File")
                open_btn = gr.Button("🌐 Preview in Browser")

            open_status = gr.Textbox(label="Status", interactive=False)

    # 绑定逻辑
    run_btn.click(
        fn=ui_to_html,
        inputs=[image_input, prompt_input],
        outputs=[model_output, download_output]
    )

    open_btn.click(
        fn=open_in_browser,
        inputs=[download_output],
        outputs=[open_status]
    )

# 启动应用
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


