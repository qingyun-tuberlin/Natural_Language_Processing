import math
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1) Tokenizer
# -----------------------------
def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)


# -----------------------------
# 2) read CSV -> list[str]
# -----------------------------
def load_texts(csv_path: str, use_columns=("Title", "Description")) -> List[str]:
    df = pd.read_csv(csv_path)
    for c in use_columns:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}. Found: {list(df.columns)}")

    text = df[use_columns[0]].fillna("").astype(str)
    for c in use_columns[1:]:
        text = text + " " + df[c].fillna("").astype(str)
    return text.tolist()


@dataclass
class Vocab:
    # String to Index
    # e.g. {"apple": 5, "banana": 6}
    stoi: dict

    # Index to String
    # e.g. ["<pad>", "<s>", "</s>", "apple", ...]
    itos: list

    # pad (<pad>): Padding space. Used to pad sentences of varying lengths to the same length for batch processing.
    pad: str = "<pad>"

    # bos (<s>): Beginning of Sentence. Tells the model to start predictions from this point.
    bos: str = "<s>"
    eos: str = "</s>"
    unk: str = "<unk>"

    @property
    def pad_id(self): return self.stoi[self.pad]
    @property
    def bos_id(self): return self.stoi[self.bos]
    @property
    def eos_id(self): return self.stoi[self.eos]
    @property
    def unk_id(self): return self.stoi[self.unk]
    @property
    def size(self): return len(self.itos)

    def encode_tokens(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in toks]

    def decode_ids(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]


def build_vocab(texts: List[str], min_freq: int = 1) -> Vocab:
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))

    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    itos = specials[:]
    for w, f in cnt.items():
        if f >= min_freq and w not in specials:
            itos.append(w)

    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


# -----------------------------
# 4) Convert the entire text into a "continuous token stream"
# Then perform next-token prediction
# -----------------------------
def texts_to_token_ids(texts: List[str], vocab: Vocab) -> List[int]:
    ids = []
    for t in texts:
        toks = [vocab.bos] + tokenize(t) + [vocab.eos]
        ids.extend(vocab.encode_tokens(toks))
    return ids


class LMSequenceDataset(Dataset):
    """
    Given a sequence of token IDs, stream:
    Take an input x of length seq_len
    Predict the sequence y of the next token (shifted right by one position)
    """
    def __init__(self, token_ids: List[int], seq_len: int = 32):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # Each sample requires seq_len+1 tokens (because we need to generate the next token).
        return max(0, len(self.data) - (self.seq_len + 1))

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]   # input
        y = chunk[1:]    # Target (shift one position to the right)
        return x, y


# -----------------------------
# 5) Neural Language Model: Embedding + (GRU/LSTM) + Linear
# -----------------------------
class NeuralLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256,
                 rnn_type: str = "gru", num_layers: int = 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [B, T]
        e = self.emb(x)            # [B, T, E]
        h, _ = self.rnn(e)         # [B, T, H]
        logits = self.fc(h)        # [B, T, V]
        return logits


# -----------------------------
# 6) Training / Evaluating perplexity / Generating
# -----------------------------
def train_neural_lm(
    train_ids: List[int],
    vocab: Vocab,
    seq_len: int = 32,
    batch_size: int = 64,
    epochs: int = 2,
    lr: float = 2e-3,
    rnn_type: str = "gru",
    device: str = "cpu",
):
    ds = LMSequenceDataset(train_ids, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = NeuralLM(vocab_size=vocab.size, rnn_type=rnn_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()  # Includes softmax + NLL (for more stable values)

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        steps = 0
        for x, y in dl:
            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B, T]
            logits = model(x) # [B, T, V]

            # The CrossEntropyLoss input requires [N, C] and [N] respectively.
            loss = crit(logits.reshape(-1, vocab.size), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        ppl = math.exp(avg_loss)
        print(f"Epoch {ep+1}/{epochs} - loss={avg_loss:.4f} - ppl={ppl:.2f}")

    train_time = time.perf_counter() - t0
    return model, train_time


@torch.no_grad()
def perplexity_neural_lm(model: nn.Module, test_ids: List[int], vocab: Vocab,
                         seq_len: int = 32, batch_size: int = 64, device: str = "cpu") -> float:
    ds = LMSequenceDataset(test_ids, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Accumulated total loss makes it easier to calculate the average.
    crit = nn.CrossEntropyLoss(reduction="sum") 
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits.reshape(-1, vocab.size), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_nll = total_loss / max(1, total_tokens)
    return math.exp(avg_nll)


@torch.no_grad()
def generate_neural_lm(model: nn.Module, vocab: Vocab, max_tokens: int = 30,
                       temperature: float = 1.0, device: str = "cpu", seed: int = 0) -> str:
    torch.manual_seed(seed)

    # Start from <s>
    context = torch.tensor([[vocab.bos_id]], dtype=torch.long, device=device)

    out_tokens = []
    model.eval()

    for _ in range(max_tokens):
        logits = model(context)              # [1, T, V]
        next_logits = logits[:, -1, :]       # [1, V]
        next_logits = next_logits / max(1e-6, temperature)

        probs = torch.softmax(next_logits, dim=-1)  # [1, V]
        next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == vocab.eos_id:
            break

        out_tokens.append(vocab.itos[next_id])

        # Append the generated token to the context (continuously increasing).
        context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)

    return " ".join(out_tokens)


device = "cuda" if torch.cuda.is_available() else "cpu"

# train_texts has 120k piece of data, even run the program overnight, I don't get result.
# Therefore, I constraint the size to first 10k
train_texts = load_texts("train.csv")[:10000] 
test_texts  = load_texts("test.csv")[:10000]

vocab = build_vocab(train_texts, min_freq=1)

train_ids = texts_to_token_ids(train_texts, vocab)
test_ids  = texts_to_token_ids(test_texts, vocab)

model, train_time = train_neural_lm(
    train_ids, vocab,
    seq_len=32, batch_size=64,
    epochs=2, lr=2e-3,
    rnn_type="gru",  # or "lstm"
    device=device
)

pp_nn = perplexity_neural_lm(model, test_ids, vocab, seq_len=32, batch_size=64, device=device)
print("Neural LM perplexity:", pp_nn)
print("Neural LM training time (s):", train_time)

print("Sample generation 1:", generate_neural_lm(model, vocab, max_tokens=25, temperature=1.0, device=device, seed=42))
print("Sample generation 2:", generate_neural_lm(model, vocab, max_tokens=25, temperature=0.8, device=device, seed=43))
