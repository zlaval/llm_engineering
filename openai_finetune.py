import os

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Electronics",
    split="full",
    trust_remote_code=True
)

print(f"Dataset loaded with {len(dataset)} samples.")
print(dataset[0])
