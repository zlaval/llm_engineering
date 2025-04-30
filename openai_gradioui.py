import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai = OpenAI()


def chat(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )

    return response.choices[0].message.content


print(chat("What is the current date?"))

gr.Interface(fn=chat,
             inputs=[gr.Textbox(lines=8, label="Prompt", placeholder="Enter a prompt")],
             outputs=[gr.Textbox(lines=8, label="Answer")],
             flagging_mode="never").launch()
