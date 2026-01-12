import math
import re
import random
from collections import Counter
import pandas as pd


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
        each file contains 3 columns, they are:
            Class Index, Title, Description
        Example Data:
            Class Index,Title,Description
            3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."

    Reading:
        For Laplace Smoothing
        https://www.geeksforgeeks.org/nlp/additive-smoothing-techniques-in-language-models/
        For Text Generation
        https://www.geeksforgeeks.org/machine-learning/text-generation-using-recurrent-long-short-term-memory-network/
        https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms
        https://towardsdatascience.com/text-generation-gpt-2-lstm-markov-chain-9ea371820e1e/


"""


class NgramLanguageModel:
    """
    Simple n-gram LM with Laplace (add-alpha) smoothing.

    P(w | context) = (count(context,w) + alpha) / (count(context) + alpha*V)
    """

    def __init__(self, n: int, alpha: float = 1.0, unk_token: str = "<unk>"):
        assert n >= 1
        self.n = n
        self.alpha = alpha
        self.unk = unk_token

        self.vocab = set()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.V = 0
        self.fitted = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        A simple tokenizer:
        - lowercase
        - keeps punctuation as separate tokens
        """
        text = (text or "").lower()
        # words with optional apostrophes, numbers, or single punctuation symbols
        return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)

    def _prepare_tokens(self, tokens: list[str]) -> list[str]:
        pads = ["<s>"] * (self.n - 1)
        return pads + tokens + ["</s>"]

    def fit(self, texts: list[str]) -> None:
        # 1) build vocabulary from training texts
        for t in texts:
            self.vocab.update(self.tokenize(t))

        # add special tokens
        self.vocab.update({"<s>", "</s>", self.unk})
        self.V = len(self.vocab)

        # 2) count ngrams + contexts
        for t in texts:
            tokens = self.tokenize(t)
            tokens = [tok if tok in self.vocab else self.unk for tok in tokens]
            sent = self._prepare_tokens(tokens)

            for i in range(self.n - 1, len(sent)):
                ngram = tuple(sent[i - self.n + 1 : i + 1])
                ctx = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[ctx] += 1

        self.fitted = True

    def prob(self, word: str, context: list[str]) -> float:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if word not in self.vocab:
            word = self.unk

        if len(context) != self.n - 1:
            raise ValueError(f"context length must be {self.n - 1} for {self.n}-gram model")

        ctx = tuple(context)
        num = self.ngram_counts[ctx + (word,)] + self.alpha
        den = self.context_counts[ctx] + self.alpha * self.V
        return num / den

    def generate(self, max_tokens: int = 30, seed: int | None = None) -> str:
        """
        Generate a short sequence by sampling from P(next | context).
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if seed is not None:
            random.seed(seed)

        context = ["<s>"] * (self.n - 1)
        out = []

        vocab_list = list(self.vocab)

        for _ in range(max_tokens):
            ctx = tuple(context)
            den = self.context_counts[ctx] + self.alpha * self.V

            weights = [
                (self.ngram_counts[ctx + (w,)] + self.alpha) / den
                for w in vocab_list
            ]
            w = random.choices(vocab_list, weights=weights, k=1)[0]

            if w == "</s>":
                break

            out.append(w)

            if self.n > 1:
                context = (context + [w])[-(self.n - 1):]

        return " ".join(out)

    def perplexity(self, texts: list[str]) -> float:
        """
        Perplexity = exp( - (1/N) * sum log P(w_i | context_i) )
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        logp_sum = 0.0
        N = 0

        for t in texts:
            tokens = self.tokenize(t)
            tokens = [tok if tok in self.vocab else self.unk for tok in tokens]
            sent = self._prepare_tokens(tokens)

            for i in range(self.n - 1, len(sent)):
                ctx = sent[i - self.n + 1 : i]
                w = sent[i]
                p = self.prob(w, ctx)
                logp_sum += math.log(p)
                N += 1

        return math.exp(-logp_sum / N) if N > 0 else float("inf")

class Task3NgramLanguageModel:
    """
    Wrapper for Task 3.1:
    - load train/test csv
    - train bigram + trigram with Laplace smoothing
    - generate samples
    - compute perplexities
    """

    def __init__(self, alpha: float = 1.0, use_columns=("Title", "Description")):
        self.alpha = alpha
        self.use_columns = use_columns
        self.bigram = NgramLanguageModel(n=2, alpha=alpha)
        self.trigram = NgramLanguageModel(n=3, alpha=alpha)

    def _load_texts(self, csv_path: str) -> list[str]:
        df = pd.read_csv(csv_path)
        for c in self.use_columns:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}. Found: {list(df.columns)}")
        text = df[self.use_columns[0]].fillna("").astype(str)
        for c in self.use_columns[1:]:
            text = text + " " + df[c].fillna("").astype(str)
        return text.tolist()

    def train(self, train_csv: str) -> None:
        train_texts = self._load_texts(train_csv)
        self.bigram.fit(train_texts)
        self.trigram.fit(train_texts)

    def evaluate(self, test_csv: str) -> dict:
        test_texts = self._load_texts(test_csv)
        pp2 = self.bigram.perplexity(test_texts)
        pp3 = self.trigram.perplexity(test_texts)
        return {"bigram_perplexity": pp2, "trigram_perplexity": pp3}

    def demo_generation(self, k: int = 3, max_tokens: int = 30, seed: int = 0) -> dict:
        bigram_samples = [self.bigram.generate(max_tokens=max_tokens, seed=seed + i) for i in range(k)]
        trigram_samples = [self.trigram.generate(max_tokens=max_tokens, seed=seed + i) for i in range(k)]
        return {"bigram": bigram_samples, "trigram": trigram_samples}



task = Task3NgramLanguageModel(alpha=1.0)  # Laplace smoothing (alpha=1)
task.train("train.csv")

print(task.demo_generation(k=3, max_tokens=25, seed=42))

results = task.evaluate("test.csv")
print(results)
