
# Medical Dialogue Privacy Pipeline

This project implements the process shown in our slide:

1. Input patient/doctor conversations
2. Clinical entity extraction / NER
3. Entity standardisation
4. K-anonymity
5. Text reconstruction
6. Anonymised conversations

## Data format

The code expects:

```text
train.json
validation.json
test.json
```

Each file should contain a list of records like:

```json
{
  "description": "...",
  "utterances": [
    "patient: ...",
    "doctor: ..."
  ]
}
```

## Run

```bash
pip install pandas
python privacy_pipeline.py --input_dir . --output_dir outputs --k 15
```

For  local folder, put `privacy_pipeline.py`, `train.json`, `validation.json`, and `test.json` in the same directory.

## Important design choice

The k-anonymity mapping is learned from `train.json` only, then applied to validation and test.  
This avoids learning privacy thresholds from evaluation data.

## Output files

```text
outputs/entities_train.csv
outputs/entities_validation.csv
outputs/entities_test.csv
outputs/entities_train_anonymized.csv
outputs/entities_validation_anonymized.csv
outputs/entities_test_anonymized.csv
outputs/k_anonymity_mapping.csv
outputs/entities_all_standardized.csv
outputs/anonymized_train.json
outputs/anonymized_validation.json
outputs/anonymized_test.json
outputs/k_anonymity_report.csv
```

## Main columns

`canonical_name` = standardised entity name  
`parent_name` = more general medical category  
`entity_type` = broad category, e.g. Sign_symptom, Disease_disorder, Medication  
`anon_term` = final replacement after k-anonymity  
`anon_level` = exact, parent, or entity_type  
`equivalence_class_size` = number of training dialogues sharing this anonymised value  

## Method

For each extracted entity:

```text
exact medical concept
    ↓ if count < k
parent medical concept
    ↓ if count < k
entity_type
```

Example:

```text
Bactrim DS
    ↓
antibiotic
    ↓
Medication
```

