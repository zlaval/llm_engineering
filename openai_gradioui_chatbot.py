import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai = OpenAI()


def chat(prompt, history):
    messages = [
        {
            "role": "system",
            "content": "You are an assistant. You response in markdown format. You are polite with the user.",
        },
    ]

    for um, am in history:
        messages.append({"role": "user", "content": um})
        messages.append({"role": "assistant", "content": am})

    messages.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ""
        yield result


gr.ChatInterface(fn=chat).launch()
