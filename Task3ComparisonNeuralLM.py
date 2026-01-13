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
# 1) Tokenizer（和你 n-gram 统一）
# -----------------------------
def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)


# -----------------------------
# 2) 读 CSV -> list[str]
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


# -----------------------------
# 3) 词表
# -----------------------------
@dataclass
class Vocab:
    # String to Index
    # e.g. {"apple": 5, "banana": 6}
    stoi: dict

    # Index to String
    # e.g. ["<pad>", "<s>", "</s>", "apple", ...]
    itos: list

    # pad (<pad>): 填充位。用于将长度不一的句子补齐到相同长度，以便进行批处理。
    pad: str = "<pad>"

    # bos (<s>): 句子开始 (Beginning of Sentence)。告诉模型预测从此开始
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
# 4) 把整段文本变成“连续 token 流”
#    然后做 next-token prediction
# -----------------------------
def texts_to_token_ids(texts: List[str], vocab: Vocab) -> List[int]:
    ids = []
    for t in texts:
        toks = [vocab.bos] + tokenize(t) + [vocab.eos]
        ids.extend(vocab.encode_tokens(toks))
    return ids


class LMSequenceDataset(Dataset):
    """
    给定一个 token id 序列 stream：
    取长度 seq_len 的输入 x
    预测后面一个 token 的序列 y（右移一位）
    """
    def __init__(self, token_ids: List[int], seq_len: int = 32):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # 每个样本需要 seq_len+1 个 token（因为要做 next token）
        return max(0, len(self.data) - (self.seq_len + 1))

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]   # 输入
        y = chunk[1:]    # 目标（右移一位）
        return x, y


# -----------------------------
# 5) 神经语言模型：Embedding + (GRU/LSTM) + Linear
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
# 6) 训练 / 评估 perplexity / 生成
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
    crit = nn.CrossEntropyLoss()  # 内含 softmax + NLL（数值更稳定）

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        steps = 0
        for x, y in dl:
            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B, T]
            logits = model(x) # [B, T, V]

            # CrossEntropyLoss 输入要求 [N, C] 和 [N]
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

    crit = nn.CrossEntropyLoss(reduction="sum")  # 累加总loss方便算平均
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

    # 从 <s> 开始
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

        # 把生成的 token append 到 context（不断增长）
        context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)

    return " ".join(out_tokens)


device = "cuda" if torch.cuda.is_available() else "cpu"

train_texts = load_texts("train.csv")
test_texts  = load_texts("test.csv")

vocab = build_vocab(train_texts, min_freq=1)

train_ids = texts_to_token_ids(train_texts, vocab)
test_ids  = texts_to_token_ids(test_texts, vocab)

model, train_time = train_neural_lm(
    train_ids, vocab,
    seq_len=32, batch_size=64,
    epochs=2, lr=2e-3,
    rnn_type="gru",  # 或 "lstm"
    device=device
)

pp_nn = perplexity_neural_lm(model, test_ids, vocab, seq_len=32, batch_size=64, device=device)
print("Neural LM perplexity:", pp_nn)
print("Neural LM training time (s):", train_time)

print("Sample generation 1:", generate_neural_lm(model, vocab, max_tokens=25, temperature=1.0, device=device, seed=42))
print("Sample generation 2:", generate_neural_lm(model, vocab, max_tokens=25, temperature=0.8, device=device, seed=43))
