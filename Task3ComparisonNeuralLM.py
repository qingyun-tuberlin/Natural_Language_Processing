
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


"""
    Task 3.2 – Comparison with a Neural Language Model
    Build a simple neural language model consisting of:

    - An embedding layer
    - One LSTM or GRU layer
    - A softmax output layer

    Then compare against n-gram models on:

    - Perplexity
    - Quality of generated text
    - And, Training time
"""
class Task3ComparisonNeuralLM():

    def __init__(self):
        pass