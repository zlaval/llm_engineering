from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"

MAX_TOKENS = 150

# Char count
MIN_LENGTH = 200
MAX_LENGTH = 5 * MAX_TOKENS


class Item:
    title: str
    price: float
    category: str
    description: str = ""

    prompt = str
    test_prompt = str
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
            self.description = content
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            tokens = tokens[:MAX_TOKENS]
            text = self.tokenizer.decode(tokens)
            self.prompt = f"How much does this product costs: {text}\n"
            self.test_prompt = self.prompt
            self.test_prompt += f"Price is _ $"
            self.prompt += f"Price is {str(self.price)} $"
            self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))
