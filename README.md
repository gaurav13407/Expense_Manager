# Personal Finance ML Analysis

An academic, notebook-first project that compares four machine-learning algorithms on a personal finance transaction dataset. The goal is to study transaction categorization, unusual spending, and monthly expense forecasting. A frontend is not required; the deliverables are reproducible analysis, metrics, visualizations, and a short report.

## Dataset

`Personal_Finance_Dataset.csv` contains 1,500 transactions from 2020-01-02 through 2024-12-29, with these columns:

- `Date`
- `Transaction Description`
- `Category`
- `Amount`
- `Type`

Initial audit found no blank fields and 1,500 unique descriptions. The descriptions appear to be generated filler, and some labels are inconsistent (for example, all `Salary` rows are typed as `Expense`; `Investment` and `Other` are typed as `Income`). Treat results as an educational experiment, not financial advice or a validated real-world model. Document these limitations in every final interpretation.

## Four algorithms and tasks

1. **Logistic Regression — category classification** using cleaned amount, transaction type, and calendar month features. Descriptions are excluded because the dataset's unique filler-like text adds no useful signal.
2. **Multinomial Naive Bayes — category classification** using the same fields and split, for a fair comparison with Logistic Regression. Both classifiers are implemented in the app and their notebooks.
3. **Isolation Forest — anomaly detection** using log transaction amount and one-hot category features. Review flagged rows; without verified anomaly labels, do not present an accuracy score as ground truth. The app includes a contamination control, and the walkthrough is `notebooks/03_isolation_forest.ipynb`.
4. **Linear Regression — monthly expense forecasting** using chronological monthly expense totals. Hold out the latest months for evaluation and report MAE/RMSE against a simple baseline. The walkthrough is `notebooks/04_linear_regression_forecast.ipynb`.

The first two methods still perform weakly because the dataset has inconsistent labels and limited category signal. Forecasting may also be weak if the generated data has no real spending trend. Reporting that honestly is part of the project.

## Suggested four-person split

- **Member 1 — Data audit and preparation:** validate schema and labels; create clean, documented data-preparation steps; maintain the shared notebook structure.
- **Member 2 — Classification:** implement Logistic Regression and Naive Bayes; report macro-F1, accuracy, and confusion matrices.
- **Member 3 — Anomaly detection:** implement Isolation Forest; select and explain a contamination setting; inspect examples and visualize flagged amounts.
- **Member 4 — Forecasting and integration:** aggregate monthly expenses, implement chronological Linear Regression evaluation, compare with a naïve baseline, and assemble final findings.

All members should review the data limitations and final results together. Keep a consistent random seed and record all preprocessing and split decisions.

## Evaluation plan

- **Classification:** use a stratified held-out split; fit scaling and one-hot encoding only on training rows; compare macro-F1, accuracy, and confusion matrices.
- **Anomaly detection:** show the chosen outlier fraction and representative flagged transactions; describe findings as candidate anomalies because the file has no anomaly ground truth.
- **Forecasting:** aggregate only expenses by calendar month; split by time (never shuffle); compare Linear Regression with a baseline that predicts the previous observed month's total; report MAE and RMSE.

## Suggested work sequence

1. Agree on the dataset caveats and finish the audit.
2. Implement each method in its own notebook or clearly separated section.
3. Use shared preprocessing conventions and avoid leakage from test data.
4. Compare results, discuss failures and limitations, and prepare a concise report/presentation.

## Setup

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create notebooks in `notebooks/` as the team begins implementation. Keep generated figures and the final report in `outputs/` and `report/` respectively.

## Folder structure

```text
Expense_Trakcer/
├── app.py                       # Streamlit demo: dashboard, filters, session-only entry, ML preview
├── Personal_Finance_Dataset.csv # Source dataset
├── requirements.txt
├── src/
│   ├── data.py                  # Loading and schema checks
│   ├── analytics.py             # Isolation Forest and forecast helpers
│   └── classification.py        # Models 1 and 2: structured-field classifiers
├── scripts/
│   └── train_models.py          # Train, evaluate, and save all four fitted models
│   └── compare_category_features.py # Compare description and structured inputs
├── models/                      # Saved fitted models (.joblib)
├── notebooks/                   # Four members' analysis notebooks
├── examples/
│   ├── sample_expenses.csv      # Small five-row upload example
│   └── budgets/                 # Five distinct synthetic budget profiles
├── outputs/                     # Evaluation summary, reports, and holdout predictions
└── report/                      # Final report and presentation
```

## Run the demo

```bash
source .venv/bin/activate
streamlit run app.py
```

The demo includes overview charts, transaction filters, a transaction-entry form that only changes the current session, and previews for Logistic Regression, Naive Bayes, Isolation Forest, and Linear Regression. Both category classifiers use cleaned amount, type, and month fields on the same stratified 80/20 holdout. In the ML preview, enter those fields for category suggestions.

Open `notebooks/01_logistic_regression.ipynb` and `notebooks/02_naive_bayes.ipynb` for the classifier walkthroughs and comparison. To run the app after activating the environment:

```bash
streamlit run app.py
```

To try CSV upload, choose any one CSV from `examples/budgets/` in the sidebar uploader, then open the **Upload CSV** tab. The five profiles range from a tight student budget to a high-spending month; their totals are listed in `examples/budgets/README.md`. Uploads need `Date`, `Transaction Description`, `Amount`, and `Type`. `Category` is optional: when supplied, the app compares model suggestions with the supplied labels; when omitted, it still adds category suggestions to the rows.

## Train and evaluate all four models

With the project environment activated, run:

```bash
python scripts/train_models.py
```

This uses `Personal_Finance_Dataset.csv`, saves four fitted models in `models/`, and writes evaluation metrics, classification reports, confusion matrices, anomaly candidate scores, and forecast holdout predictions to `outputs/`. The anomaly detector has no label-based accuracy score because this dataset contains no verified anomaly labels. The evaluation summary records the library version used to create the saved models.

To compare category-classification inputs on the same held-out split, run:

```bash
python scripts/compare_category_features.py
```

This writes `outputs/category_feature_experiment.csv`. In the current dataset, description-only accuracy is about 10.7%; amount, transaction type, and month raise it to about 19–20%. That remains weak. Using transaction type alone reaches about 20% accuracy but only 0.091 macro F1, because type is strongly entangled with category labels (for example, the dataset marks every `Salary` row as an expense). The app's category models use structured fields and let you try amount, type, and month. Do not use accuracy alone to claim the model is good.
# Expense_Manager
