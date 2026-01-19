"""
Task 5: Human Evaluation Analysis

This script analyzes human evaluation results stored in
'human_eval_merged.csv', where two annotators independently
judge whether model predictions are reasonable.

Expected columns in CSV:
- text
- model_prediction
- human_1
- human_2

Human ratings should be one of:
- Correct
- Partially correct
- Incorrect
"""

import pandas as pd


# =========================
# Configuration
# =========================

HUMAN_EVAL_PATH = "human_eval_merged.csv"


# =========================
# Utility Functions
# =========================

def map_human_rating(rating):
    """
    Map human rating to binary acceptance.
    Correct / Partially correct -> 1
    Incorrect -> 0
    """
    if rating in ["correct", "partially correct"]:
        return 1
    return 0


# =========================
# Main Analysis
# =========================

if __name__ == "__main__":

    # -------- Load data --------
    df = pd.read_csv(HUMAN_EVAL_PATH)

    print("Loaded human evaluation file")
    print(df.head(), "\n")

    # -------- Map human ratings to binary --------
    df["human_1_bin"] = df["human_1"].apply(map_human_rating)
    df["human_2_bin"] = df["human_2"].apply(map_human_rating)

    # -------- Majority / acceptance rule --------
    # Accepted if at least one annotator finds it reasonable
    df["human_accept"] = (df["human_1_bin"] + df["human_2_bin"]) >= 1

    # -------- Basic statistics --------
    total = len(df)
    accept_rate = df["human_accept"].mean()

    print("Human Evaluation Summary")
    print(f"Total evaluated samples: {total}")
    print(f"Human acceptance rate: {accept_rate:.3f}\n")

    # -------- Annotator agreement --------
    agreement = (df["human_1_bin"] == df["human_2_bin"]).mean()

    print("Annotator Agreement")
    print(f"Exact agreement rate between annotators: {agreement:.3f}\n")

    # -------- Distribution of ratings --------
    print("Human 1 rating distribution:")
    print(df["human_1"].value_counts(), "\n")

    print("Human 2 rating distribution:")
    print(df["human_2"].value_counts(), "\n")

    # -------- Disagreement cases --------
    disagreements = df[df["human_1_bin"] != df["human_2_bin"]]

    print(f"Number of disagreement cases: {len(disagreements)}")

    if len(disagreements) > 0:
        print("\nSample disagreement cases:")
        print(
            disagreements[
                ["text", "model_prediction", "human_1", "human_2"]
            ].head(5)
        )

    print("\nAnalysis complete.")
