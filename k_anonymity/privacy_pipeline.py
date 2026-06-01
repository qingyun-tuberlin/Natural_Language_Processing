
"""
End-to-end privacy pipeline for medical dialogue data.

Input:
    train.json
    validation.json
    test.json

Output:
    outputs/
        entities_train.csv
        entities_validation.csv
        entities_test.csv
        entities_all_standardized.csv
        k_anonymity_mapping.csv
        anonymized_train.json
        anonymized_validation.json
        anonymized_test.json
        k_anonymity_report.csv

Design:
    1. NER: rule/dictionary based medical entity extraction.
    2. Entity standardisation: map surface terms to canonical medical concepts.
    3. K-anonymity: learn generalisation mapping on TRAIN only.
    4. Reconstruction: replace original mentions by anonymised terms.

This version is deliberately offline and reproducible.
You can later replace the NER/standardisation layer with scispaCy + UMLS/SNOMED.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# 1. Medical lexicon
# ---------------------------------------------------------------------
# term -> (canonical_id, canonical_name, parent_id, parent_name, entity_type)
#
# In a stronger version, canonical_id would be SNOMED CT ID or UMLS CUI.
# Here we use readable local IDs so the code runs without external licenses.
LEXICON: Dict[str, Tuple[str, str, str, str, str]] = {
    # symptoms
    "fever": ("SYM_FEVER", "fever", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "low grade fever": ("SYM_FEVER", "fever", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "high fever": ("SYM_FEVER", "fever", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "temperature": ("SYM_FEVER", "fever", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "temp": ("SYM_FEVER", "fever", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "cough": ("SYM_COUGH", "cough", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "dry cough": ("SYM_DRY_COUGH", "dry cough", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "persistent cough": ("SYM_COUGH", "cough", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "shortness of breath": ("SYM_DYSPNEA", "shortness of breath", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "sob": ("SYM_DYSPNEA", "shortness of breath", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "difficulty breathing": ("SYM_DYSPNEA", "shortness of breath", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "breathing difficulty": ("SYM_DYSPNEA", "shortness of breath", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "wheezing": ("SYM_WHEEZE", "wheezing", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "sore throat": ("SYM_SORE_THROAT", "sore throat", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "throat pain": ("SYM_SORE_THROAT", "sore throat", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "scratchy throat": ("SYM_SORE_THROAT", "sore throat", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "headache": ("SYM_HEADACHE", "headache", "SYM_NEURO", "neurological symptom", "Sign_symptom"),
    "headaches": ("SYM_HEADACHE", "headache", "SYM_NEURO", "neurological symptom", "Sign_symptom"),
    "dizzy": ("SYM_DIZZINESS", "dizziness", "SYM_NEURO", "neurological symptom", "Sign_symptom"),
    "dizziness": ("SYM_DIZZINESS", "dizziness", "SYM_NEURO", "neurological symptom", "Sign_symptom"),
    "tiredness": ("SYM_FATIGUE", "fatigue", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "fatigue": ("SYM_FATIGUE", "fatigue", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "weak": ("SYM_WEAKNESS", "weakness", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "weakness": ("SYM_WEAKNESS", "weakness", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "chills": ("SYM_CHILLS", "chills", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "muscle aches": ("SYM_MYALGIA", "muscle aches", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "body aches": ("SYM_MYALGIA", "muscle aches", "SYM_SYSTEMIC", "systemic symptom", "Sign_symptom"),
    "chest pain": ("SYM_CHEST_PAIN", "chest pain", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "tight chest": ("SYM_CHEST_TIGHTNESS", "chest tightness", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "chest tightness": ("SYM_CHEST_TIGHTNESS", "chest tightness", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "runny nose": ("SYM_RUNNY_NOSE", "runny nose", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "stuffy nose": ("SYM_NASAL_CONGESTION", "nasal congestion", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "nasal congestion": ("SYM_NASAL_CONGESTION", "nasal congestion", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "sneezing": ("SYM_SNEEZING", "sneezing", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "phlegm": ("SYM_SPUTUM", "sputum", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "sputum": ("SYM_SPUTUM", "sputum", "SYM_RESP", "respiratory symptom", "Sign_symptom"),
    "vomit": ("SYM_VOMITING", "vomiting", "SYM_GI", "gastrointestinal symptom", "Sign_symptom"),
    "vomiting": ("SYM_VOMITING", "vomiting", "SYM_GI", "gastrointestinal symptom", "Sign_symptom"),
    "nauseous": ("SYM_NAUSEA", "nausea", "SYM_GI", "gastrointestinal symptom", "Sign_symptom"),
    "nausea": ("SYM_NAUSEA", "nausea", "SYM_GI", "gastrointestinal symptom", "Sign_symptom"),
    "diarrhea": ("SYM_DIARRHEA", "diarrhea", "SYM_GI", "gastrointestinal symptom", "Sign_symptom"),
    "diaree": ("SYM_DIARRHEA", "diarrhea", "SYM_GI", "gastrointestinal symptom", "Sign_symptom"),

    # diseases/disorders
    "covid": ("DIS_COVID19", "COVID-19", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "covid-19": ("DIS_COVID19", "COVID-19", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "covid19": ("DIS_COVID19", "COVID-19", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "coronavirus": ("DIS_COVID19", "COVID-19", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "corona virus": ("DIS_COVID19", "COVID-19", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "sars-cov-2": ("DIS_COVID19", "COVID-19", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "pneumonia": ("DIS_PNEUMONIA", "pneumonia", "DIS_RESP", "respiratory disease", "Disease_disorder"),
    "walking pneumonia": ("DIS_ATYPICAL_PNEUMONIA", "atypical pneumonia", "DIS_PNEUMONIA_GROUP", "pneumonia", "Disease_disorder"),
    "bronchitis": ("DIS_BRONCHITIS", "bronchitis", "DIS_RESP", "respiratory disease", "Disease_disorder"),
    "sinus infection": ("DIS_SINUSITIS", "sinusitis", "DIS_RESP", "respiratory disease", "Disease_disorder"),
    "sinusitis": ("DIS_SINUSITIS", "sinusitis", "DIS_RESP", "respiratory disease", "Disease_disorder"),
    "strep throat": ("DIS_STREP_THROAT", "streptococcal sore throat", "DIS_RESP", "respiratory disease", "Disease_disorder"),
    "cold": ("DIS_COMMON_COLD", "common cold", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "common cold": ("DIS_COMMON_COLD", "common cold", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "flu": ("DIS_INFLUENZA", "influenza", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "influenza": ("DIS_INFLUENZA", "influenza", "DIS_VIRAL_RESP", "viral respiratory infection", "Disease_disorder"),
    "asthma": ("DIS_ASTHMA", "asthma", "DIS_RESP", "respiratory disease", "Disease_disorder"),
    "diabetes": ("DIS_DIABETES", "diabetes mellitus", "DIS_ENDOCRINE", "endocrine disease", "Disease_disorder"),
    "hypertension": ("DIS_HYPERTENSION", "hypertension", "DIS_CARDIO", "cardiovascular disease", "Disease_disorder"),
    "high blood pressure": ("DIS_HYPERTENSION", "hypertension", "DIS_CARDIO", "cardiovascular disease", "Disease_disorder"),
    "cancer": ("DIS_CANCER", "cancer", "DIS_NEOPLASM", "neoplastic disease", "Disease_disorder"),
    "infection": ("DIS_INFECTION", "infection", "DIS_INFECTIOUS", "infectious disease", "Disease_disorder"),
    "infections": ("DIS_INFECTION", "infection", "DIS_INFECTIOUS", "infectious disease", "Disease_disorder"),
    "allergy": ("DIS_ALLERGY", "allergy", "DIS_IMMUNE", "immune system disorder", "Disease_disorder"),
    "anxiety": ("DIS_ANXIETY", "anxiety disorder", "DIS_MENTAL", "mental disorder", "Disease_disorder"),

    # medications/treatments
    "antibiotic": ("MED_ANTIBIOTIC", "antibiotic", "MED_ANTIINFECTIVE", "anti-infective medication", "Medication"),
    "antibiotics": ("MED_ANTIBIOTIC", "antibiotic", "MED_ANTIINFECTIVE", "anti-infective medication", "Medication"),
    "bactrim ds": ("MED_BACTRIM_DS", "Bactrim DS", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "azithromycin": ("MED_AZITHRO", "azithromycin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "azithromiacin": ("MED_AZITHRO", "azithromycin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "amoxicillin": ("MED_AMOXICILLIN", "amoxicillin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "amoxillin": ("MED_AMOXICILLIN", "amoxicillin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "erythromycin": ("MED_ERYTHRO", "erythromycin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "clindamycin": ("MED_CLINDA", "clindamycin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "levaquin": ("MED_LEVOFLOXACIN", "levofloxacin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "klacid xl": ("MED_CLARITHRO", "clarithromycin", "MED_ANTIBIOTIC", "antibiotic", "Medication"),
    "steroids": ("MED_STEROID", "steroid", "MED_ANTIINFLAMMATORY", "anti-inflammatory medication", "Medication"),
    "prednisone": ("MED_PREDNISONE", "prednisone", "MED_STEROID", "steroid", "Medication"),
    "ibuprofen": ("MED_IBUPROFEN", "ibuprofen", "MED_NSAID", "NSAID", "Medication"),
    "nsaids": ("MED_NSAID", "NSAID", "MED_ANTIINFLAMMATORY", "anti-inflammatory medication", "Medication"),
    "acetaminophen": ("MED_ACETAMINOPHEN", "acetaminophen", "MED_ANALGESIC", "analgesic medication", "Medication"),
    "paracetamol": ("MED_ACETAMINOPHEN", "acetaminophen", "MED_ANALGESIC", "analgesic medication", "Medication"),
    "tylenol": ("MED_ACETAMINOPHEN", "acetaminophen", "MED_ANALGESIC", "analgesic medication", "Medication"),
    "vaccine": ("TRT_VACCINE", "vaccine", "TRT_PREVENTIVE", "preventive treatment", "Treatment"),
    "vaccination": ("TRT_VACCINE", "vaccine", "TRT_PREVENTIVE", "preventive treatment", "Treatment"),
    "vaccinations": ("TRT_VACCINE", "vaccine", "TRT_PREVENTIVE", "preventive treatment", "Treatment"),
    "flu vaccine": ("TRT_FLU_VACCINE", "influenza vaccine", "TRT_VACCINE", "vaccine", "Treatment"),
    "pneumonia shot": ("TRT_PNEUMO_VACCINE", "pneumococcal vaccine", "TRT_VACCINE", "vaccine", "Treatment"),
    "pneumococcal vaccine": ("TRT_PNEUMO_VACCINE", "pneumococcal vaccine", "TRT_VACCINE", "vaccine", "Treatment"),
    "inhaler": ("TRT_INHALER", "inhaler", "TRT_RESP_SUPPORT", "respiratory treatment", "Treatment"),
    "inhalers": ("TRT_INHALER", "inhaler", "TRT_RESP_SUPPORT", "respiratory treatment", "Treatment"),
}


@dataclass(frozen=True)
class Entity:
    dialogue_id: str
    split: str
    utterance_index: int
    speaker: str
    start: int
    end: int
    original_term: str
    canonical_id: str
    canonical_name: str
    parent_id: str
    parent_name: str
    entity_type: str


def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dialogue_text(item: dict) -> str:
    return "\n".join(item.get("utterances", []))


def speaker_of_utterance(utt: str) -> str:
    low = utt.lower().strip()
    if low.startswith("patient:"):
        return "patient"
    if low.startswith("doctor:"):
        return "doctor"
    return "unknown"


# Longest terms first prevents "cough" from stealing "dry cough".
SORTED_TERMS = sorted(LEXICON.keys(), key=len, reverse=True)
TERM_PATTERN = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(t) for t in SORTED_TERMS) + r")(?![a-zA-Z])",
    flags=re.IGNORECASE,
)


def extract_entities_from_utterance(
    text: str,
    dialogue_id: str,
    split: str,
    utterance_index: int,
) -> List[Entity]:
    entities: List[Entity] = []
    speaker = speaker_of_utterance(text)

    occupied: List[Tuple[int, int]] = []

    for match in TERM_PATTERN.finditer(text):
        start, end = match.span()
        # Avoid overlapping matches.
        if any(not (end <= s or start >= e) for s, e in occupied):
            continue
        occupied.append((start, end))

        original = match.group(0)
        key = original.lower()
        canonical_id, canonical_name, parent_id, parent_name, entity_type = LEXICON[key]

        entities.append(
            Entity(
                dialogue_id=dialogue_id,
                split=split,
                utterance_index=utterance_index,
                speaker=speaker,
                start=start,
                end=end,
                original_term=original,
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                parent_id=parent_id,
                parent_name=parent_name,
                entity_type=entity_type,
            )
        )

    return entities


def extract_entities(data: list, split: str) -> pd.DataFrame:
    rows: List[dict] = []

    for i, item in enumerate(data):
        dialogue_id = f"{split}_{i}"
        for j, utt in enumerate(item.get("utterances", [])):
            for ent in extract_entities_from_utterance(utt, dialogue_id, split, j):
                rows.append(ent.__dict__)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. K-anonymity
# ---------------------------------------------------------------------
def learn_k_anonymity_mapping(train_entities: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Learn mapping from canonical concept -> anonymised concept using TRAIN only.

    Generalisation path:
        exact canonical concept
        -> parent concept
        -> entity type

    Equivalence class size = number of distinct dialogues in train split.
    """
    if train_entities.empty:
        raise ValueError("No entities extracted from training data.")

    base = (
        train_entities[
            ["canonical_id", "canonical_name", "parent_id", "parent_name", "entity_type", "dialogue_id"]
        ]
        .drop_duplicates()
        .copy()
    )

    concept_table = (
        base[["canonical_id", "canonical_name", "parent_id", "parent_name", "entity_type"]]
        .drop_duplicates("canonical_id")
        .copy()
    )

    exact_counts = (
        base.groupby("canonical_id")["dialogue_id"]
        .nunique()
        .rename("exact_count")
        .reset_index()
    )

    parent_counts = (
        base.groupby(["entity_type", "parent_id"])["dialogue_id"]
        .nunique()
        .rename("parent_count")
        .reset_index()
    )

    type_counts = (
        base.groupby("entity_type")["dialogue_id"]
        .nunique()
        .rename("type_count")
        .reset_index()
    )

    mapping = concept_table.merge(exact_counts, on="canonical_id", how="left")
    mapping = mapping.merge(parent_counts, on=["entity_type", "parent_id"], how="left")
    mapping = mapping.merge(type_counts, on="entity_type", how="left")

    anon_ids = []
    anon_terms = []
    anon_levels = []
    sizes = []

    for _, row in mapping.iterrows():
        if row["exact_count"] >= k:
            anon_ids.append(row["canonical_id"])
            anon_terms.append(row["canonical_name"])
            anon_levels.append("exact")
            sizes.append(int(row["exact_count"]))
        elif row["parent_count"] >= k:
            anon_ids.append(row["parent_id"])
            anon_terms.append(row["parent_name"])
            anon_levels.append("parent")
            sizes.append(int(row["parent_count"]))
        else:
            anon_ids.append(f"ENTITY_TYPE::{row['entity_type']}")
            anon_terms.append(row["entity_type"])
            anon_levels.append("entity_type")
            sizes.append(int(row["type_count"]))

    mapping["anon_id"] = anon_ids
    mapping["anon_term"] = anon_terms
    mapping["anon_level"] = anon_levels
    mapping["equivalence_class_size"] = sizes
    mapping["k"] = k
    mapping["is_k_anonymous"] = mapping["equivalence_class_size"] >= k

    return mapping.sort_values(["anon_level", "entity_type", "canonical_name"])


def apply_mapping(entities: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "canonical_id",
        "anon_id",
        "anon_term",
        "anon_level",
        "equivalence_class_size",
        "is_k_anonymous",
        "k",
    ]
    return entities.merge(mapping[keep], on="canonical_id", how="left")


def reconstruct_dialogues(
    data: list,
    split: str,
    anonymized_entities: pd.DataFrame,
) -> list:
    """
    Replace entity mentions by anon_term.

    Replacements are done from right to left in each utterance so character offsets remain valid.
    """
    out = json.loads(json.dumps(data))  # deep copy

    if anonymized_entities.empty:
        return out

    ent_by_dialogue = {
        did: group.copy()
        for did, group in anonymized_entities.groupby("dialogue_id")
    }

    for i, item in enumerate(out):
        did = f"{split}_{i}"
        group = ent_by_dialogue.get(did)
        if group is None:
            continue

        for utt_idx in sorted(group["utterance_index"].unique()):
            rows = group[group["utterance_index"] == utt_idx].sort_values("start", ascending=False)
            text = item["utterances"][int(utt_idx)]
            for _, row in rows.iterrows():
                replacement = str(row["anon_term"])
                text = text[: int(row["start"])] + replacement + text[int(row["end"]) :]
            item["utterances"][int(utt_idx)] = text

        item["description"] = " ".join(
            utt.split(":", 1)[-1].strip() if ":" in utt else utt
            for utt in item.get("utterances", [])
        )

    return out


def report(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for level, group in mapping.groupby("anon_level"):
        rows.append({
            "metric": f"concepts_with_anon_level={level}",
            "value": len(group),
        })

    rows.append({"metric": "total_concepts", "value": len(mapping)})
    rows.append({"metric": "all_concepts_k_anonymous", "value": bool(mapping["is_k_anonymous"].all())})

    return pd.DataFrame(rows)


def run_pipeline(input_dir: Path, output_dir: Path, k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": load_json(input_dir / "train.json"),
        "validation": load_json(input_dir / "validation.json"),
        "test": load_json(input_dir / "test.json"),
    }

    entity_frames = {}
    for split, data in datasets.items():
        ents = extract_entities(data, split)
        entity_frames[split] = ents
        ents.to_csv(output_dir / f"entities_{split}.csv", index=False)

    train_entities = entity_frames["train"]
    mapping = learn_k_anonymity_mapping(train_entities, k=k)
    mapping.to_csv(output_dir / "k_anonymity_mapping.csv", index=False)

    standardized_all = []

    for split, ents in entity_frames.items():
        anon_ents = apply_mapping(ents, mapping)
        anon_ents.to_csv(output_dir / f"entities_{split}_anonymized.csv", index=False)
        standardized_all.append(anon_ents)

        anonymized_dialogues = reconstruct_dialogues(datasets[split], split, anon_ents)
        with (output_dir / f"anonymized_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(anonymized_dialogues, f, ensure_ascii=False, indent=2)

    all_df = pd.concat(standardized_all, ignore_index=True)
    all_df.to_csv(output_dir / "entities_all_standardized.csv", index=False)

    rep = report(mapping)
    rep.to_csv(output_dir / "k_anonymity_report.csv", index=False)

    print("Done.")
    print(f"K = {k}")
    print(f"Output directory: {output_dir}")
    print()
    print("K-anonymity mapping summary:")
    print(mapping["anon_level"].value_counts())
    print()
    print("Report:")
    print(rep.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_dir, args.k)


if __name__ == "__main__":
    main()
