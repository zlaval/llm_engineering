
import os
import re
import math
from tqdm import tqdm
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datetime import datetime
from dotenv import load_dotenv

BASE_MODEL = "meta-llama/Llama-3.2-1B"
FINE_TUNED_MODEL = "zlaval/price-calc-2025-05-25-1"

# Hyperparameters for QLoRA Fine-Tuning
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")

print(base_model)









