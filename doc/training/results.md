# Four-model training results

This report summarizes the latest run of the four independent training scripts. The source scores are recorded in `outputs/model_comparison.csv`.

## Evaluation setup

- **Target:** transaction-line quantity. Training quantities were capped at 1,000 and transformed with `log1p`; evaluation predictions are converted back to quantity units.
- **Inputs:** product code, country, unit price, and year/month/day/weekday/hour derived from invoice date.
- **Time split:** model training uses transactions before November 2011. The holdout covers **November 1, 2011 through December 9, 2011**.
- **Sample size:** each script uses a seeded sample of up to 60,000 training rows and 20,000 holdout rows. This run used those maximum counts for each model.
- **Comparison:** all four scripts use the same split and seed, making the scores comparable on this evaluation sample.

## Scores

| Model | MAE (lower is better) | RMSE (lower is better) | R² (higher is better) |
| --- | ---: | ---: | ---: |
| Histogram Gradient Boosting | 6.248 | 23.021 | 0.112 |
| Random Forest | 6.273 | **22.590** | **0.145** |
| Extra Trees | 6.701 | 23.617 | 0.065 |
| Ridge Regression | 7.310 | 24.630 | -0.017 |

## What the scores say

- **Random Forest** has the lowest RMSE and highest R² in this run. It is the strongest overall by those two measures.
- **Histogram Gradient Boosting** has the lowest MAE, by about 0.025 quantity units compared with Random Forest. Their average absolute errors are very close.
- **Extra Trees** performs better than Ridge on all three reported metrics, but falls behind Random Forest and Histogram Gradient Boosting.
- **Ridge Regression** has negative R², meaning it performs worse than a constant-average prediction under the R² comparison for this holdout.
- The R² values are relatively low, so the current inputs explain only a limited share of the variation in quantity. The scores are useful for comparing these four implementations, but prediction quality can be improved.

## How to read the metrics

- **MAE** is the average absolute prediction error, in item-quantity units. For example, an MAE of about 6.25 means predictions differ from the capped actual quantity by about 6.25 items on average.
- **RMSE** is also an error in item-quantity units, but penalizes larger misses more heavily.
- **R²** compares model predictions with a constant-mean baseline. A value closer to 1 is better; 0 is similar to that baseline, and a negative value is worse than it.

## Saved files

- `models/ridge_regression.joblib`
- `models/random_forest.joblib`
- `models/extra_trees.joblib`
- `models/histogram_gradient_boosting.joblib`
- `outputs/ridge_regression_predictions.csv`
- `outputs/random_forest_predictions.csv`
- `outputs/extra_trees_predictions.csv`
- `outputs/histogram_gradient_boosting_predictions.csv`
- `outputs/model_comparison.csv`

These results describe one chronological holdout and seeded sample. They are not a guarantee of future model performance. Also, categorical codes are currently ordinal-encoded, even though product codes and countries have no natural numeric order; that encoding is a clear candidate for improvement before treating these as final results.
