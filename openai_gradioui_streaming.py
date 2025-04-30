import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai = OpenAI()


def chat(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an assistant. You response in markdown format."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ""
        yield result


print(chat("What is the current date?"))

gr.Interface(fn=chat,
             inputs=[gr.Textbox(lines=20, label="Prompt", placeholder="Enter a prompt")],
             outputs=[gr.Markdown( label="Answer")],
             flagging_mode="never").launch()
