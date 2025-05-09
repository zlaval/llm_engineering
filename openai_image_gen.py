import base64
import gradio as gr
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI()


def generate(prompt):
    response = openai.images.generate(
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json",
        model="dall-e-3",
        n=1,
    )
    image_b64 = response.data[0].b64_json
    image_data = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_data))
    # img = Image.open(BytesIO(image_data))
    # img.save("image.png")


# generate("A futuristic city skyline at sunset, with flying cars and neon lights")

with gr.Blocks() as ui:
    with gr.Row():
        image = gr.Image(label="Image", height=400)
    with gr.Row():
        msg = gr.Textbox(label="Message", placeholder="Enter a message")

    msg.submit(fn=generate, inputs=msg, outputs=image)

ui.launch()
