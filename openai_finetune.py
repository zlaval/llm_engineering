import json
import os
from collections import defaultdict
from typing import Optional

#import matplotlib.pyplot as plt
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_TOKENS = 150

# Char count
MIN_LENGTH = 200
MAX_LENGTH = 5 * MAX_TOKENS

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Appliances",
    split="full",  # code will generate full split
    trust_remote_code=True  # enable dataset scripts, which do some processing when download the dataset
)

# remove datapoints without price
dataset = dataset.filter(lambda x: x["price"] not in ["None", "", None])

example_datapoint = dataset[0]
#print(f"Datapoints number: {len(dataset)}")
#print(json.dumps(example_datapoint, indent=4))
#print("Details")

#print(example_datapoint['title'])
#print(example_datapoint['price'])
#print(example_datapoint['details'])
#print(example_datapoint['features'])

#prices = [round(float(datapoint["price"])) for datapoint in dataset]

## Too much cheap items (avg/mean) not good training data,
#plt.figure(figsize=(15, 6))
#plt.title("Prices")
#plt.xlabel("Price")
#plt.ylabel("Count")
#plt.hist(prices, rwidth=0.7, bins=range(0, 1000, 10))
#plt.show()


## dataset = dataset.select(range(30000)).filter(lambda x: x["price"] not in ["None", "", None]).select(range(2000))


class Item:
    title: str
    price: float
    category: str
    description: str

    prompt = Optional[str]
    test_prompt = Optional[str]
    include = False

    token_count: int = 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def __init__(self, data):
        self.title = data['title']
        self.price = float(data['price'])
        self.category = data['main_category']
        self.parse(data)

    def __str__(self):
        return (f"Item(title={self.title}, price={self.price}, category={self.category},"
                f"prompt={self.prompt}, include={self.include}, token_count={self.token_count})")

    def __repr__(self):
        return self.__str__()

    def parse(self, data):
        content = '\n'.join(data['description'])
        content += '\n'.join(data['features'])
        content += '\n'
        content += data['details']  # json

        if len(content) > MIN_LENGTH:
            self.include = True
            content = content[:MAX_LENGTH]
            # We should cleanup the content
            content = f"{self.title}\n{content}"
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            tokens = tokens[:MAX_TOKENS]
            text = self.tokenizer.decode(tokens)
            self.prompt = f"How much does this product costs: {text}\n"
            self.test_prompt=self.prompt
            self.test_prompt += f"Price is _ $"
            self.prompt += f"Price is {str(self.price)} $"
            self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))


print("Create datapoints")

MIN_PRICE = 1
MAX_PRICE = 1000

items = []
for datapoint in dataset:
    p = float(datapoint['price'])
    if MIN_PRICE <= p <= MAX_PRICE:
        item = Item(datapoint)
        items.append(item)

print(f"Total items: {len(items)}")
print(items[0])

#Key is the rounded price and value is a list of items has that price
slots = defaultdict(list)
for item in items:
    slots[round(item.price)].append(item)

#Create balanced sample from the data





# TODO average price, balanced price distribution
