import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import torch
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from transformers import AutoTokenizer
# print("torch version:", torch.__version__)
# print("mps available:", torch.backends.mps.is_available())
# print("mps built:", torch.backends.mps.is_built())


""" 
Task 3.1 – N-gram Language Models
For this task you should implement a bigram and a trigram language model
Requirements:

- Compute smoothed probabilities (e.g., Laplace smoothing)
- Generate short text sequences from each model
- Compute and compare perplexity across n-gram sizes

Files:
    train.csv
    test.csv
"""
class Task3NgramLanguageModel():
    def __init__(self):
        pass




