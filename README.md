# SecondLook

**An AI-assisted second review for borderline loan declines.**
Group 3 capstone, Uplift Code Camp, Python for Data and AI (Batch 3).

## Problem

About half of Filipino adults have no formal financial account, and thin credit files get many of them declined with no explanation.

**Business question:** Among applicants a lender would decline, which ones are actually safe to approve, and what realistic changes would get the rest approved?

## What SecondLook does

1. Scores each applicant's default risk with a trained model.
2. Flags borderline declines near the approval cutoff.
3. Explains each score with SHAP.
4. Suggests a realistic path to approval (never age, gender or past defaults).
5. Writes a plain-language decision letter with Claude, then checks it against the model.
6. Answers portfolio questions in plain English through an AI agent.

## Data

[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk/data): 307,511 labelled applicants (8.07% default rate) plus 6 credit-history tables.

The data is not stored in this repo. To download it:

```bash
pip install kaggle
kaggle auth login            # or put your token in ~/.kaggle/
kaggle competitions download home-credit-default-risk -p data/raw
```

Keep the zip as is. The code reads CSVs straight from it (`src/secondlook/data.py`).

## Project structure

```
data/raw/          Kaggle zip (not committed)
data/interim/      cleaned tables, SQLite DB (not committed)
data/processed/    model-ready features (not committed)
notebooks/         01 EDA, 02 features, 03 modelling, ...
src/secondlook/    reusable Python package (config, data loading, features, models)
app/               Streamlit app (Home.py + pages/)
models/            trained model files (not committed)
reports/figures/   charts used in the README and pitch
docs/              proposal and model card
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

Run the app (once built):

```bash
streamlit run app/Home.py
```

## Roadmap

- **Phase 1, by Nov 26 (mock):** EDA, features, models, SHAP, Claude letters, Streamlit MVP.
- **Phase 2, by Dec 2 (final):** path to approval, faithfulness and tone checks, fairness audit, personas, AI Analyst agent.

## Team

Group 3. Roles to be filled in.
