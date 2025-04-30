from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI()


def generate(text):
    response = openai.audio.speech.create(
        input=text,
        voice="alloy",
        model="tts-1",
    )

    audio = BytesIO(response.content)
    with open("audio.mp3", "wb") as f:
        f.write(audio.getbuffer())


generate("Hello llm developer. How are you doing today?")
