# Histogram Gradient Boosting Regressor

## What we built

The Histogram Gradient Boosting script is `scripts/training/train_hist_gradient_boosting.py`. It builds decision trees sequentially. Each new tree focuses on reducing errors left by the earlier trees. It groups numeric values into bins (histograms) to make split search efficient.

## Why we included it

Boosting provides a different tree-ensemble strategy from Random Forest and Extra Trees. It can build a strong model by adding small corrections in sequence and gives us another way to capture nonlinear retail patterns.

## Advantages

- Learns nonlinear effects and feature interactions.
- Histogram-based split search can be efficient on larger datasets.
- Sequential corrections can fit patterns that a simple linear baseline misses.
- L2 regularization is included to help control model complexity.

## Trade-offs

- Training is sequential, so it is less naturally parallel across trees than a forest.
- It can overfit if allowed too many iterations or overly complex trees.
- Like the other tree models, it has limited ability to extrapolate beyond the training range.
- The current ordinal encoding of product and country is a limitation shared with the other models.

## Shared training choices

The script allows up to 150 boosting iterations, limits trees to 31 leaves, uses a fixed random seed, and shares the chronological holdout and preprocessing. It is evaluated with MAE, RMSE, and R². The fitted pipeline is saved as `models/histogram_gradient_boosting.joblib`.
