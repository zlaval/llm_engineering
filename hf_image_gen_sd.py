import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline

if not torch.cuda.is_available():
    raise Exception("CUDA not available")

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


def generate(prompt):
    image = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=4.5,
        max_sequence_length=512,
    ).images[0]
    return image


with gr.Blocks() as ui:
    with gr.Row():
        image_panel = gr.Image(label="Image", height=500)
    with gr.Row():
        msg = gr.Textbox(label="Message", placeholder="Enter a message")
    msg.submit(fn=generate, inputs=msg, outputs=image_panel)

ui.launch()
