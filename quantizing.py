import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

LLM = "google/gemma-3-1b-it"

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}, ]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a joke for a programmer"}, ]
        },
    ],
]

qc = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Normalized Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(LLM, quantization_config=qc)

m = model.get_memory_footprint() / 1e6
print(f"Model memory footprint: {m:.2f} MB")
print(model)

# pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)
# output = pipe(messages, max_new_tokens=80)

tokenizer = AutoTokenizer.from_pretrained(LLM)
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
output = model.generate(inputs, max_new_tokens=80)
print(tokenizer.decode(output[0]))
