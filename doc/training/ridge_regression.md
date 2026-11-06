# Ridge Regression

## What we built

The Ridge script is `scripts/training/train_ridge.py`. It fits a linear regression model with L2 regularization to predict transaction-line quantity. Its inputs are product code, country, unit price, and calendar fields derived from the invoice date.

## Why we included it

Ridge gives us a simple, fast baseline. It helps answer whether a relatively simple relationship between the prepared inputs and quantity can explain the data before using more complex tree models.

## Advantages

- Trains quickly and is inexpensive to save and run.
- Regularization limits overly large coefficients and can make a linear model more stable.
- Gives us a useful baseline against which to judge the tree-based models.

## Trade-offs

- It learns a mostly linear relationship and can miss complex interactions, such as a product behaving differently by country or season.
- The categorical encoder maps product and country names to integer codes. Those codes have no natural order, but a linear model can treat them as ordered values. This is a known limitation of the current shared preprocessing and a candidate for improving later.
- Quantity predictions can be less accurate when the underlying patterns are strongly nonlinear.

## Shared training choices

The script uses the same final-two-month chronological holdout, seeded row limits, preprocessing pipeline, log-transformed quantity target, and MAE/RMSE/R² metrics as the other models. This makes the comparison more consistent. The fitted pipeline is saved as `models/ridge_regression.joblib`.
