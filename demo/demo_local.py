#!/usr/bin/env python3
"""
Local Gradio Demo for UI2Codeⁿ — Visual UI-to-Code Model

Run:
    python ui2code_demo_local.py
"""

import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# ============================================================
# 1️⃣ Model Configuration
# ============================================================

MODEL_ID = "./ui2code"  # 改成你自己的模型名（比如本地路径或 HF 模型名）
# MODEL_ID = "zai-org/UI2Code_N"  # 改成你自己的模型名（比如本地路径或 HF 模型名）

print(f"🔧 Loading model from {MODEL_ID} ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("✅ Model loaded successfully!")


# ============================================================
# 2️⃣ Inference Function
# ============================================================

def generate_ui_code(image, prompt):
    """
    Given a UI screenshot and prompt, generate HTML/CSS code.
    """
    if image is None:
        return "⚠️ Please upload a UI image first."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt or "Please generate the HTML/CSS code for this UI screenshot."}
        ]
    }]

    # Prepare model input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return output_text


# ============================================================
# 3️⃣ Gradio Interface
# ============================================================

demo = gr.Interface(
    fn=generate_ui_code,
    inputs=[
        gr.Image(type="filepath", label="Upload UI Screenshot"),
        gr.Textbox(label="Prompt", value="Please generate the HTML code for this UI."),
    ],
    outputs=gr.Code(label="Generated HTML Code", language="html"),
    title="UI2Codeⁿ: Visual UI-to-Code Model",
    description=(
        "🧠 <b>UI2Codeⁿ</b> is a visual language model that unifies UI-to-code generation, "
        "UI editing, and UI polishing. Upload a UI screenshot and generate its corresponding HTML/CSS."
    ),
    examples=[
        ["assets/example.png", "Generate HTML for this UI screenshot."],
    ],
)

# ============================================================
# 4️⃣ Run Locally
# ============================================================

if __name__ == "__main__":
    print("🚀 Launching Gradio demo at http://127.0.0.1:7880 ...")
    demo.launch(server_name="0.0.0.0", server_port=7880, share=False)

