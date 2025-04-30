import base64
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
    image_b64=response.data[0].b64_json
    image_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_data))
    img.save("image.png")


generate("A futuristic city skyline at sunset, with flying cars and neon lights")
