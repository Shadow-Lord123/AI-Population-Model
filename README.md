# AI-Population-Model

This is for my final 80% marks Artificial Intelligence and Machine Learning model.

## Predicting Population Decline with Supervised Learning

CMP6202 Artificial Intelligence and Machine Learning Project (2026–27) · Student ID 23187763

This project predicts whether a country's population is **shrinking** (annual growth rate < 0) in a given year from its demographic profile. It compares **Logistic Regression**, a **Decision Tree** and a **Random Forest**, using a **train/test split by year** (1955–2003 train, 2004–2023 test) to prevent temporal leakage.

## Results (test set, 2004–2023)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.453 | 0.910 | 0.605 | 0.907 |
| Decision Tree | 0.505 | 0.576 | 0.538 | 0.879 |
| **Random Forest** | **0.681** | **0.670** | **0.676** | **0.932** |

## Repository contents

```
population_decline_model.ipynb   executed notebook (all outputs and charts)
population_decline_model.py      the same pipeline as a plain script
data/WPP2024_Demographic_Indicators_Medium.csv.gz   raw dataset (UN WPP 2024)
data/population_decline_model_data.csv   the 13,869 cleaned rows used for modelling (with Train/Test split column)
figures/                         charts used in the report
results.json                     all metrics written by the notebook
requirements.txt
```

## How to run

```bash
git clone https://github.com/Shadow-Lord123/AI-Population-Model.git
cd AI-Population-Model
pip install -r requirements.txt
jupyter notebook population_decline_model.ipynb    # or: python population_decline_model.py
```

Run from the repository root so the relative `data/` and `figures/` paths resolve. A full run takes about 2 minutes on a laptop. Results are reproducible (`random_state=42`).

## Method in brief

1. Keep Country/Area rows only, estimates only (1950–2023), and drop micro-states with a peak population below 100,000 → 201 countries.
2. Target: `Declining = PopGrowthRate < 0`. Remove every column that arithmetically reconstructs growth (population totals, births, deaths, crude rates, natural change, same-year migration).
3. Engineered features using past data only: `CNMR_prev5`, `TFR_change5`, `MedianAge_change5`, `LEx_gap`.
4. Expanding-window cross-validation on whole years; grid search scored on F1; class weighting for imbalance.

## Data source and licence

United Nations, Department of Economic and Social Affairs, Population Division (2024). *World Population Prospects 2024*, Online Edition. https://population.un.org/wpp/ — licensed under **CC BY 3.0 IGO** (http://creativecommons.org/licenses/by/3.0/igo/).
