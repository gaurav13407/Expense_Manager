# UK Household Spending Trends and Forecasts

A four-model machine-learning project using the UK Living Costs and Food Survey workbook. It prepares category-level spending history, compares four regression algorithms, and provides a small Streamlit dashboard for exploring historical averages and forecasts.

## Project question

How has average weekly household spending changed across major categories, and how do four regression algorithms forecast the next few periods from that history?

## Data

The source workbook is stored at `data/raw/workbook1detailedexpenditureandtrends.xlsx`. The project reads:

- Worksheet `4.1`: average weekly household spending by category across the historical periods. The values are reported at 2024–25 prices.
- Worksheet `A6`: latest-year spending comparisons by gross-income decile, shown in the dashboard for context only.

The prepared file `data/processed/category_spending.csv` has one row per category and source period. The source is a survey of household groups, not transaction-level or individual-household records. Forecasts are demonstrations for a coursework project, not financial advice or precise official forecasts.

## Four algorithms and team split

Each person can own the algorithm implementation in `src/models/`:

1. **Linear Regression** — simple baseline using period, category and recent spending.
2. **Random Forest Regressor** — tree ensemble that can model nonlinear patterns.
3. **Gradient Boosting Regressor** — sequential trees with robust Huber loss.
4. **Support Vector Regressor (SVR)** — nonlinear regression with an RBF kernel.

All models share the same features and chronological evaluation so their scores can be compared fairly. Features include category, period order, previous-period spending, second-previous spending, and the mean of the previous three observations.

## Run the project

From the project folder, install dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Prepare the cleaned data and train/evaluate all four models:

```bash
python scripts/prepare_dataset.py
python scripts/train_models.py
```

The training script evaluates the last four periods using a rolling one-step-ahead split, then refits each model on all available history and saves the fitted models. It writes:

- `models/`: four saved model pipelines.
- `outputs/model_comparison.csv`: MAE, RMSE and R² for each model.
- `outputs/holdout_predictions.csv`: actual and predicted holdout values.
- `outputs/three_period_forecasts.csv`: next three period estimates from each model.

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard lets users select a spending category and model, inspect its historical and forecast values, compare evaluation errors, and view current-year income-decile comparisons.

## Evaluation and limitations

- The holdout split is chronological; later periods are not shuffled into training data.
- MAE and RMSE are in pounds of average weekly household spending. Lower values mean smaller errors. R² is supplementary and can be negative.
- The holdout is small: only four periods per category, so model rankings can change with a different split.
- Category-level survey averages smooth over variation between households. The model cannot categorize an individual expense or identify an individual household's unusual purchase.
- Historical data includes changes in survey periods and unusual pandemic-era spending. Use the dashboard for learning and comparison, and explain those caveats in any presentation.

## Folder structure

```text
Expense_Trakcer/
├── app.py
├── data/
│   ├── raw/                  # Supplied survey workbook
│   └── processed/            # Tidy category-by-period CSV
├── models/                   # Trained model pipelines
├── outputs/                  # Holdout evaluation and forecast CSVs
├── scripts/
│   ├── prepare_dataset.py
│   └── train_models.py
└── src/
    ├── data.py               # Workbook loading and reshaping
    ├── forecasting.py        # Shared features, training, evaluation, forecasts
    └── models/               # One algorithm module per team member
```
