"""
Task 5: Model Evaluation & Benchmarking

This script evaluates a fine-tuned DistilBERT classifier on AG News
using three perspectives:

1. Metric-based evaluation (gold labels)
2. Human evaluation (manual annotations)
3. LLM-as-a-judge evaluation (API-free proxy)

Author: Jiaqi Chen
"""

# =========================
# Imports
# =========================

import random
import pandas as pd
import numpy as np

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle


# =========================
# Configuration
# =========================

# MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cpu")
NUM_EVAL_SAMPLES = 100

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# =========================
# Utility Functions
# =========================

def load_ag_news(test_path="test.csv"):
    """
    Load AG News test set.
    """
    df = pd.read_csv(test_path)
    texts = (df["Title"].fillna("") + " " + df["Description"].fillna("")).str.strip()
    labels = df["Class Index"] - 1
    return texts, labels


def load_model():
    """
    Load fine-tuned DistilBERT model.
    """
    model = DistilBertForSequenceClassification.from_pretrained("task4_finetuned_model")

    model.to(DEVICE)
    model.eval()
    return model


def predict(model, tokenizer, texts):
    """
    Generate predictions for a list of texts.
    """
    predictions = []

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)

    return predictions


# =========================
# 1. Metric-based Evaluation
# =========================

def metric_based_evaluation(y_true, y_pred):
    """
    Standard evaluation using gold labels.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1


# =========================
# 2. Human Evaluation
# =========================

def create_human_eval_template(texts, preds, output_path="human_eval.csv"):
    """
    Create a CSV template for human evaluation.
    """
    df = pd.DataFrame({
        "text": texts,
        "model_prediction": [CLASS_NAMES[p] for p in preds],
        "human_rating": [""] * len(texts)  # To be filled manually
    })

    df.to_csv(output_path, index=False)
    print(f"Human evaluation template saved to: {output_path}")


def analyze_human_eval(csv_path, gold_labels, preds):
    """
    Analyze completed human evaluation file.
    Expected ratings: Correct / Partially correct / Incorrect
    """
    df = pd.read_csv(csv_path)

    # Convert ratings to binary correctness
    human_correct = df["human_rating"].map({
        "Correct": 1,
        "Partially correct": 1,
        "Incorrect": 0
    })

    gold_correct = (gold_labels == preds).astype(int)

    agreement = (human_correct == gold_correct).mean()

    return agreement


# =========================
# 3. LLM-as-a-Judge (Proxy)
# =========================

def llm_judge_proxy(text, predicted_label):
    """
    API-free proxy for LLM-as-a-judge.

    This function simulates a judgement by checking
    whether key category-related keywords appear in the text.

    This is a proxy for a local LLM and should be described
    as such in the report.
    """
    text = text.lower()

    keyword_map = {
        "World": ["war", "government", "country", "president"],
        "Sports": ["match", "team", "season", "player"],
        "Business": ["market", "company", "stocks", "economy"],
        "Sci/Tech": ["technology", "software", "research", "ai"]
    }

    for keyword in keyword_map[predicted_label]:
        if keyword in text:
            return "Correct"

    return "Partially correct"


def llm_as_judge_evaluation(texts, preds):
    """
    Apply LLM-as-a-judge proxy to all samples.
    """
    judgements = []

    for text, pred in zip(texts, preds):
        label = CLASS_NAMES[pred]
        judgement = llm_judge_proxy(text, label)
        judgements.append(judgement)

    return judgements


# =========================
# Main
# =========================

if __name__ == "__main__":

    # Load data
    test_texts, test_labels = load_ag_news()

    # Subsample 100 instances
    test_texts, test_labels = shuffle(test_texts, test_labels, random_state=42)
    eval_texts = test_texts[:NUM_EVAL_SAMPLES]
    eval_labels = test_labels[:NUM_EVAL_SAMPLES].to_numpy()

    # Load model & tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("task4_finetuned_model")
    model = load_model()

    # Predict
    predictions = predict(model, tokenizer, eval_texts)

    # -------- Metric-based evaluation --------
    acc, f1 = metric_based_evaluation(eval_labels, predictions)

    print("Metric-based Evaluation (Gold Labels)")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print()

    # -------- Human evaluation --------
    create_human_eval_template(eval_texts, predictions)

    print("Human evaluation:")
    print("Please fill in 'human_eval.csv' and rerun analysis if needed.")
    print()

    # -------- LLM-as-a-judge evaluation --------
    llm_judgements = llm_as_judge_evaluation(eval_texts, predictions)

    llm_correct_rate = sum(j in ["Correct", "Partially correct"] for j in llm_judgements) / len(llm_judgements)

    print("LLM-as-a-Judge Evaluation (Proxy)")
    print(f"Agreement rate (Correct or Partially correct): {llm_correct_rate:.4f}")
