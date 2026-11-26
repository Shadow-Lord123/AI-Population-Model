# AI-Population-Model

This is for my final 80% marks Artificial Intelligence and Machine Learning model.

## Predicting Population Growth and Decline with Regression

CMP6202 Artificial Intelligence and Machine Learning Project (2026–27) 
· Student ID 23187763

This project predicts a **number**: a country's **population growth rate** in a given year (**% per year**) from its demographic profile. A negative prediction means the population is expected to **shrink**.

It follows the module's "A to Z" workflow (Weeks 3, 5 and 8): problem → load → clean → features → split → baseline → cross-validation → tuning (`GridSearchCV`, `RandomizedSearchCV`) → final test once → explain.

The data is split **by year** (learn from 1955–2003, test on 2004–2023) and cross-validated on blocks of years, because a random split would put neighbouring years of the same country in both sets (data leakage).

## Results on the test years (2004–2023)

Errors are in **percentage points of growth** (0.5 = half a percentage point off). Lower MAE and RMSE, and higher R², are better.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 1.309 | 1.755 | -0.166 |
| Linear Regression | 0.524 | 1.055 | 0.579 |
| Decision Tree (tuned) | 0.605 | 1.237 | 0.421 |
| **Random Forest (tuned)** | **0.493** | 1.079 | 0.559 |

The tuned Random Forest has the lowest typical error. Linear Regression is close behind, with the best RMSE and R², and is easier to explain.

## Repository contents

| Path | What it is |
|---|---|
| `population_growth_model.ipynb` | Executed notebook: all outputs, charts and explanations |
| `population_growth_model.py` | The same pipeline as a plain script |
| `data/WPP2024_Demographic_Indicators_Medium.csv.gz` | Raw dataset (UN WPP 2024) |
| `data/population_growth_model_data.csv` | The 13,869 cleaned rows used for modelling |
| `figures/` | Every chart used in the report |
| `report/CMP6202_Individual_Report_23187763.docx` | The report |
| `results.json` | All scores written by the notebook |
| `requirements.txt` | Python packages needed |

## How to run

```bash
git clone https://github.com/Shadow-Lord123/AI-Population-Model.git
cd AI-Population-Model
pip install -r requirements.txt
jupyter notebook population_growth_model.ipynb    # or: python population_growth_model.py
```

Run from the repository root so `DATA_PATH = "data/"` resolves.

**In Google Colab:** upload the `data` folder to your Drive and change `DATA_PATH` to that folder, as in the module labs.

A full run takes about 10 minutes on a 2-core machine, mostly for the hyperparameter searches. Results are reproducible (`random_state=42`).

## Data source and licence

United Nations, Department of Economic and Social Affairs, Population Division (2024). *World Population Prospects 2024*, Online Edition. https://population.un.org/wpp/ (accessed 13 November 2026).

Licensed under **CC BY 3.0 IGO** (http://creativecommons.org/licenses/by/3.0/igo/), which permits academic use with attribution.
