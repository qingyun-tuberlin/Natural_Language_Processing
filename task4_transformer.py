"""
Task 4: Transformer Fine-tuning on AG News
Model: DistilBERT (CPU-only, minimal training version)

This script implements:
- Data loading and splitting
- Two tokenization options
- Dataset and DataLoader
- DistilBERT fine-tuning (1–2 epochs)
- Validation evaluation

Author: Jiaqi Chen
"""

# =========================
# Imports
# =========================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import shuffle


# =========================
# Reproducibility
# =========================

def set_seed(seed: int = 42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Data Loading
# =========================

def load_ag_news(train_path: str, test_path: str):
    """
    Load AG News dataset from CSV files.

    Expected columns:
    - Class Index (1-4)
    - Title
    - Description
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_texts = (train_df["Title"].fillna("") + " " +
                   train_df["Description"].fillna("")).str.strip()
    test_texts = (test_df["Title"].fillna("") + " " +
                  test_df["Description"].fillna("")).str.strip()

    # Convert labels from 1-4 to 0-3
    train_labels = train_df["Class Index"] - 1
    test_labels = test_df["Class Index"] - 1

    return train_texts, train_labels, test_texts, test_labels


# =========================
# Train / Validation Split
# =========================

def split_train_validation(texts, labels, val_size=0.15):
    """
    Split training data into train and validation sets
    using stratified sampling.
    """
    return train_test_split(
        texts,
        labels,
        test_size=val_size,
        random_state=42,
        stratify=labels
    )


# =========================
# Dataset Definition
# =========================

class AGNewsDataset(Dataset):
    """
    PyTorch Dataset for AG News classification.
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =========================
# Tokenizers
# =========================

def get_wordpiece_tokenizer():
    """
    Default WordPiece tokenizer used by DistilBERT.
    """
    return DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )


def get_bpe_tokenizer():
    """
    Alternative subword tokenization (fast tokenizer backend).
    Used for comparison with WordPiece.
    """
    return DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased",
        use_fast=True
    )


# =========================
# DataLoader Utility
# =========================

def build_dataloader(texts, labels, tokenizer, batch_size=16, shuffle=False):
    """
    Build DataLoader for AG News.
    Batch size is kept small for CPU training.
    """
    dataset = AGNewsDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =========================
# Training & Evaluation
# =========================

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation or test set.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1

def evaluate_with_confusion_matrix(model, dataloader, device):
    """
    Evaluate model and return accuracy, macro-F1, and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1, cm

# =========================
# Main
# =========================

if __name__ == "__main__":

    set_seed(42)
    device = torch.device("cpu")

    # -------- Data --------
    train_texts, train_labels, test_texts, test_labels = load_ag_news(
        "train.csv", "test.csv"
    )

    X_train, X_val, y_train, y_val = split_train_validation(
        train_texts, train_labels
    )
    
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)

    X_train = X_train[:3000]
    y_train = y_train[:3000]

    X_val = X_val[:500]
    y_val = y_val[:500]
    # -------- Tokenizer --------
    tokenizer_type = "wordpiece"  # change to "bpe" for comparison

    if tokenizer_type == "wordpiece":
        tokenizer = get_wordpiece_tokenizer()
    else:
        tokenizer = get_bpe_tokenizer()

    # -------- DataLoaders --------
    train_loader = build_dataloader(
        X_train, y_train, tokenizer, batch_size=16, shuffle=True
    )
    val_loader = build_dataloader(
        X_val, y_val, tokenizer, batch_size=16
    )
    test_loader = build_dataloader(
        test_texts, test_labels, tokenizer, batch_size=16
    )

    # -------- Model --------
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # -------- Training (Minimal Version) --------
    num_epochs = 2

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_acc, val_f1 = evaluate(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

    # # -------- Final Test Evaluation --------
    # test_acc, test_f1 = evaluate(model, test_loader, device)
    # print("\nTest Results:")
    # print(f"Accuracy: {test_acc:.4f}")
    # print(f"Macro-F1: {test_f1:.4f}")
    
    # -------- Final Test Evaluation + Confusion Matrix --------
    test_acc, test_f1, cm = evaluate_with_confusion_matrix(
        model, test_loader, device
    )

    print("\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")

    # -------- Confusion Matrix Visualization --------
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on AG News Test Set (wordpiece)")
    plt.tight_layout()

    # Save figure for report usage
    plt.savefig("confusion_matrix_wordpiece.png")
    plt.show()
    
    
    model.save_pretrained("task4_finetuned_model")
    tokenizer.save_pretrained("task4_finetuned_model")


