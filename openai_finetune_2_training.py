import math
import os
import random
import re

from matplotlib import pyplot as plt
from openai import OpenAI

from item import Item

_ = Item

# import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pickle
from huggingface_hub import login

MODEL_NAME = "meta-llama/Llama-3.2-1B"

BLUE = '\033[94m'
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "yellow": YELLOW, "blue": BLUE}

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)

with open('assets/train.pkl', 'rb') as file:
    train = pickle.load(file)

with open('assets/test.pkl', 'rb') as file:
    test = pickle.load(file)


class Tester:

    def __init__(self, predictor, data, title=None, size=25):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    @classmethod
    def color_for(cls, error, truth):
        if error < 40 or error / truth < 0.2:
            return "blue"
        elif error < 80 or error / truth < 0.4:
            return "yellow"
        else:
            return "red"

    def chart(self, title):
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color == "green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits / self.size * 100:.1f}%"
        self.chart(title)

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(
            f"{COLOR_MAP[color]}{i + 1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()


random.seed(42)


# def random_price(item):
#    return random.randrange(1, 1000)


def gpt_message(item):
    system_prompt = "You must estimate the price of items. Reply only with price as a plain number. No explanation."
    user_prompt = item.test_prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"},
    ]


openai = OpenAI()


def extract_price(s):
    s = s.replace('$', '').replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0


def gpt_price(item):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=gpt_message(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return extract_price(reply)


# = gpt_price(test[0])
# print(res)
#Tester.test(gpt_price, test[:30])


#https://platform.openai.com/docs/guides/supervised-fine-tuning