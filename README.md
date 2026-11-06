# Online Retail II: preprocessing and four-model training

This project prepares the supplied `Data/online_retail_II.xlsx` workbook and compares four regression models on transaction item quantity.

For a plain-language explanation of the cleaning steps and dataset counts, see [`doc/Preprocessing.md`](doc/Preprocessing.md).
Each model also has its own explanation under [`doc/training/`](doc/training/).
The latest comparison is summarized in [`doc/training/results.md`](doc/training/results.md).

## Dataset and target

The workbook contains two sheets, covering December 2009 through December 2011. Each row is an invoice line with an invoice number, product, quantity, timestamp, unit price, customer ID, and country. The training target is `Quantity`. Models use product code, country, unit price, and calendar fields derived from `InvoiceDate`. Invoice and customer IDs are not model inputs.

Preprocessing removes rows without required fields, duplicate rows, cancellations and returns (non-positive quantities), and zero or negative prices. It adds line revenue for analysis. This is transaction-level data, so predictions estimate item quantity on a line; they do not represent household spending or an individual customer's future purchases.

## Run

Install dependencies and run the pipeline from the project root:

```bash
python -m pip install -r requirements.txt
python scripts/preprocessing/prepare_dataset.py
python scripts/training/train_ridge.py
python scripts/training/train_random_forest.py
python scripts/training/train_extra_trees.py
python scripts/training/train_hist_gradient_boosting.py
```

Run preprocessing once before training. Each model file contains its own complete training pipeline, so a teammate can work on that model directly without relying on a shared training helper. The four training commands can be run independently, in any order. Each command uses the same final-two-month chronological holdout and seeded sample limits (up to 60,000 training rows and 20,000 evaluation rows), then saves its fitted pipeline and holdout predictions. Each run updates its model's row in `outputs/model_comparison.csv`.

## Models

1. **Ridge Regression** — fast, regularized linear baseline. See [`doc/training/ridge_regression.md`](doc/training/ridge_regression.md).
2. **Random Forest Regressor** — averages trees to learn nonlinear patterns. See [`doc/training/random_forest.md`](doc/training/random_forest.md).
3. **Extra Trees Regressor** — uses extra-randomized tree splits for comparison. See [`doc/training/extra_trees.md`](doc/training/extra_trees.md).
4. **Histogram Gradient Boosting Regressor** — adds trees sequentially to reduce earlier errors. See [`doc/training/histogram_gradient_boosting.md`](doc/training/histogram_gradient_boosting.md).

All models use the same chronological split and inputs. Categorical fields are encoded with an unknown-category fallback; numeric fields are imputed and scaled. The quantity target is capped at 1,000 and log-transformed during fitting to reduce the influence of extreme quantities. Evaluation reports MAE, RMSE, and R² after predictions are converted back to quantity units.

## Output files

- `outputs/model_comparison.csv`: holdout metrics and split details for each model.
- `outputs/ridge_regression_predictions.csv`, `outputs/random_forest_predictions.csv`, `outputs/extra_trees_predictions.csv`, and `outputs/histogram_gradient_boosting_predictions.csv`: actual and predicted quantity for each model's holdout rows.
- `models/*.joblib`: fitted scikit-learn pipelines, ready for inference with the same features.

The scores should be interpreted as a comparison on this dataset and split, not as guarantees for future transactions. The last two months are a relatively small time window, and unseen products or seasonal patterns may perform differently.
