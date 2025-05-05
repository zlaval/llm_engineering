import os
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_TOKENS = 1000
MAX_LENGTH = 5 * MAX_TOKENS

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Electronics",
    split="full",
    trust_remote_code=True
)

dataset = dataset.select(range(30000)).filter(lambda x: x["price"] not in ["None", "", None]).select(range(2000))

print(f"Dataset loaded with {len(dataset)} samples.")
print(dataset[0])


class Item:
    title: str
    price: float
    category: str
    average_rating: float
    rating_number: int

    prompt = Optional[str]
    include = False

    token_count: int = 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def __init__(self, data):
        self.title = data['title']
        self.price = data['price']
        self.category = data['main_category']
        self.average_rating = data['average_rating']
        self.rating_number = data['rating_number']

        self.parse(data)

    def __str__(self):
        return (f"Item(title={self.title}, price={self.price}, category={self.category}, "
                f"average_rating={self.average_rating}, rating_number={self.rating_number}, "
                f"prompt={self.prompt}, include={self.include}, token_count={self.token_count})")

    def __repr__(self):
        return self.__str__()

    def parse(self, data):
        content = '\n'.join(data['description'])
        content += '\n'.join(data['features'])
        content += '\n'.join(data['details'])  # json

        if len(content) > 0:
            content = content[:MAX_LENGTH]
            content = f"{self.title}\n{content}"
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            tokens = tokens[:MAX_TOKENS]
            text = self.tokenizer.decode(tokens)
            self.include = True
            self.prompt = f"How much does this product costs: {text}\n"
            self.prompt += f"Price is {str(self.price)}"
            self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

print("Create datapoints")
items= []
for datapoint in dataset:
    item = Item(datapoint)
    items.append(item)

print(f"Total items: {len(items)}")
print(items[0])