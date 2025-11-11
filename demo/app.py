import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "zai-org/UI2Code_N"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

def generate_code(image, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    }]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    output = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return output

demo = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Image(type="filepath", label="Upload UI Image"),
        gr.Textbox(label="Prompt", value="Generate HTML for this UI")
    ],
    outputs=gr.Code(label="Generated Code", language="html"),
    title="UI2Codeⁿ Demo",
    description="A Visual Language Model for Interactive UI-to-Code Generation."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

