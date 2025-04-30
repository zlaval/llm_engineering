import os

import torch
import gradio as gr
from diffusers import FluxPipeline
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")

login(token=api_key)

if not torch.cuda.is_available():
    raise Exception("CUDA not available")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate(prompt):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]
    return image


with gr.Blocks() as ui:
    with gr.Row():
        image_panel = gr.Image(label="Image", height=600)
    with gr.Row():
        msg = gr.Textbox(label="Message", placeholder="Enter a message")
    msg.submit(fn=generate, inputs=msg, outputs=image_panel)

ui.launch()
