import os
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
# import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from matplotlib import pyplot as plt

from item import Item

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

# print(f"Datapoints number: {len(dataset)}")
# print(json.dumps(example_datapoint, indent=4))
# print("Details")

# print(example_datapoint['title'])
# print(example_datapoint['price'])
# print(example_datapoint['details'])
# print(example_datapoint['features'])

# prices = [round(float(datapoint["price"])) for datapoint in dataset]

## Too much cheap items (avg/mean) not good training data,
# plt.figure(figsize=(15, 6))
# plt.title("Prices")
# plt.xlabel("Price")
# plt.ylabel("Count")
# plt.hist(prices, rwidth=0.7, bins=range(0, 1000, 10))
# plt.show()


## dataset = dataset.select(range(30000)).filter(lambda x: x["price"] not in ["None", "", None]).select(range(2000))


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

# Key is the rounded price and value is a list of items has that price
slots = defaultdict(list)
for item in items:
    if item.include:
        slots[round(item.price)].append(item)

# Create a balanced sample from the data
sample = []
for i in range(MIN_PRICE, MAX_PRICE):
    slot = slots[i]
    # as we have just a few item worth more than the value, we take those all
    if i >= 150:
        sample.extend(slot)
    # also take all when there are less than x elements for a given price
    elif len(slot) < 50:
        sample.extend(slot)
    else:
        weights = np.array([2 if len(item.description) > 100 else 1 for item in slot])
        weights = weights / np.sum(weights)
        # select x elements randomly from a slot, longer text bigger probability (return indices)
        si = np.random.choice(len(slot), size=50, replace=False, p=weights)
        s = [slot[j] for j in si]
        sample.extend(s)

print(f"number of samples: {len(sample)}")

prices = [round(float(item.price)) for item in sample]
plt.figure(figsize=(15, 6))
plt.title(f"Prices. Avg {sum(prices) / len(prices):.2f}. Highest {max(prices):.2f}\n")
plt.xlabel("Price")
plt.ylabel("Count")
plt.hist(prices, rwidth=0.7, bins=range(0, 1000, 10))  # step 10, it will be summed
plt.show()


# TODO average price, balanced price distribution
def report(item):
    prompt = item.prompt
    tokens = item.tokenizer.encode(prompt, add_special_tokens=False)
    print(prompt)
    # last 10 tokens
    print(tokens[-10:])
    print(item.tokenizer.batch_decode(tokens[-10:]))


report(sample[100])

# mixup sample data
random.seed(42)
random.shuffle(sample)
length = len(sample)
test_len = int(length / 10)
train_length = length - test_len
# 90% train 10% test
train = sample[:train_length]
test = sample[train_length:]

print(f"Train sample: {train[0].prompt}")
# test prompt does not contain price so llm must figure it
print(f"Test sample: {test[0].test_prompt}")

# create final dataset
training_prompts = [item.prompt for item in train]
training_prices = [item.price for item in train]
test_prompts = [item.prompt for item in test]
test_prices = [item.price for item in test]

train_dataset = Dataset.from_dict({"text": training_prompts, "price": training_prices})
test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# HF_USER = "zlaval"
# DATASET_NAME = f"{HF_USER}/price-data"
# dataset.push_to_hub(DATASET_NAME, private=True)
#
# with open('assets/train.pkl', 'wb') as file:
#     pickle.dump(train, file)
#
# with open('assets/test.pkl', 'wb') as file:
#     pickle.dump(test, file)
