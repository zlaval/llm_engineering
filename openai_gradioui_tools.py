import json

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai = OpenAI()

car_prices = {
    "toyota": "24000",
    "honda": "26000",
    "ford": "32000",
    "chevrolet": "23000",
    "bmw": "41000",
    "tesla": "39000",
}


def get_price(car_model):
    car = car_model.lower()
    return car_prices.get(car, "I dont know")


price_fn = {
    "name": "get_price",
    "description": "Get the price of a car model",
    "parameters": {
        "type": "object",
        "properties": {
            "car_model": {
                "type": "string",
                "description": "The car model to get the price for",
            },
        },
        "required": ["car_model"],
        "additionalProperties": False,
    },
}

tools = [{
    "type": "function",
    "function": price_fn
}]


def chat(prompt, history):
    messages = [
        {
            "role": "system",
            "content": "You are a polite assistant of a car seller company called Carcamp. Give short, exact answers in one sentence. Always be accurate, if you dont know the answer, say 'I dont know'.",
        },
    ]

    for um, am in history:
        messages.append({"role": "user", "content": um})
        messages.append({"role": "assistant", "content": am})
    messages.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    if response.choices[0].finish_reason == "tool_calls":
        msg = response.choices[0].message
        resp, price = handle_tool_call(msg)
        messages.append(msg)
        messages.append(resp)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

    return response.choices[0].message.content or ""


def handle_tool_call(msg):
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    car = args.get("car_model")
    price = get_price(car)
    response = {
        "role": "tool",
        "content": f"The price of {car} is {price}",
        "tool_call_id": tool_call.id,
    }
    return response, car


gr.ChatInterface(fn=chat).launch()
