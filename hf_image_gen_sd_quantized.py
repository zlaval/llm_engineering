import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from transformers import BitsAndBytesConfig

if not torch.cuda.is_available():
    raise Exception("CUDA not available")

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
).to("cuda")


def generate(prompt):
    image = pipeline(
        prompt,
        num_inference_steps=20,
        guidance_scale=4.5,
        max_sequence_length=512,
    ).images[0]
    return image


with gr.Blocks() as ui:
    with gr.Row():
        image_panel = gr.Image(label="Image", height=600)
    with gr.Row():
        msg = gr.Textbox(label="Message", placeholder="Enter a message")
    msg.submit(fn=generate, inputs=msg, outputs=image_panel)

ui.launch()
