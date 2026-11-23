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
notebooks/         01 EDA, 02 features, 03 modelling, 04 explainability, 05 letters
src/secondlook/    reusable Python package: data.py (load from zip), cleaning.py,
                   sql_features.py (SQLite + SQL aggregations), features.py (pipeline),
                   modeling.py, explain.py (SHAP), letters.py (Claude), demo_bundle.py
app/               Streamlit app (Home.py + pages/), demo_data/ bundle, requirements.txt
tests/             pytest tests (letter guardrails, app smoke tests)
models/            trained model files (not committed)
reports/figures/   charts used in the README and pitch
docs/              proposal, feature dictionary, model card
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

Build the model-ready feature table (about 10 minutes the first time; creates a 2.3 GB SQLite database in `data/interim/` and `data/processed/features.parquet`):

```bash
cd src && python -m secondlook.features
```

Or run `notebooks/02_features.ipynb`, which does the same step by step with explanations.

## Run the app

The app runs on a small committed bundle in `app/demo_data/` (model, 61,503 test predictions, 300 applicants, pre-drafted letters), so **it works straight after cloning**, without the Kaggle data:

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

| Page | What it shows |
|---|---|
| **Home** | Problem, headline results, how SecondLook works |
| **Portfolio Dashboard** | Risk distribution, live cutoff and profit simulator, calibration, approvals vs score-only, thin-file results |
| **Applicant Review** | One applicant: PD, decision, SHAP reasons, Claude letter (pre-drafted, or drafted live with an API key), officer decision |
| **Model Card** | Model details, metrics, SHAP importance, fairness check, intended use and limitations |

Live letter drafting needs `ANTHROPIC_API_KEY` in `.env` (locally) or in the app's secrets (cloud). Without a key, everything else works and pre-drafted letters are shown.

To rebuild the bundle after retraining (needs the full pipeline): `cd src && python -m secondlook.demo_bundle`

### Deploy to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → repository `Funkchini/secondlook`, branch `main`, main file `app/Home.py`.
3. **Advanced settings** → Python 3.12, and under **Secrets** add `ANTHROPIC_API_KEY = "..."` (optional, for live letters).
4. Deploy. The app installs `app/requirements.txt`.

## Tests

```bash
pytest tests/
```

24 tests: letter guardrail rules (16) and app smoke tests that run every page, the cutoff slider and the cached letters (8). No API calls.

## Roadmap

- **Phase 1, by Nov 26 (mock):** EDA, features, models, SHAP, Claude letters, Streamlit MVP.
- **Phase 2, by Dec 2 (final):** path to approval, faithfulness and tone checks, fairness audit, personas, AI Analyst agent.

## Team

Group 3. Roles to be filled in.
