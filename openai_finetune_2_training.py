import os
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
# import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

MODEL_NAME = "meta-llama/Llama-3.2-1B"

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")
login(token=api_key)


# with open('assets/train.pkl', 'wb') as file:
#     pickle.dump(train, file)
#
# with open('assets/test.pkl', 'wb') as file:
#     pickle.dump(test, file)
