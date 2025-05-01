import os

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")

login(token=api_key)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

text = "Hello, I'm a test sentence. Supercalifragilisticexpialidocious. I'd like to order the chicken sandwich, please. My brother is my best friend; we do everything together."
tokens = tokenizer.encode(text)

print(tokens)

token_text = tokenizer.decode(tokens)
print(token_text)

token_btext = tokenizer.batch_decode(tokens)
print(token_btext)
