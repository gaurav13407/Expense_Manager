# Random Forest Regressor

## What we built

The Random Forest script is `scripts/training/train_random_forest.py`. It trains an ensemble of decision trees. Each tree learns quantity patterns from a slightly different sample and feature subset; their predictions are averaged.

## Why we included it

Retail quantities can depend on combinations of product, price, country, and timing. A forest can learn nonlinear patterns and interactions without requiring us to specify those relationships by hand.

## Advantages

- Can model nonlinear relationships and feature interactions.
- Averaging many trees usually makes results more stable than relying on one tree.
- Needs less numeric scaling sensitivity than a linear model.
- Provides a strong, easy-to-explain tree-ensemble comparison point.

## Trade-offs

- Larger and slower to fit and save than Ridge Regression.
- A forest can struggle to extrapolate beyond patterns seen in its training data.
- The current integer encoding of product and country categories gives arbitrary codes an order in tree splits; a different categorical encoding may improve this.

## Shared training choices

The script uses 50 trees, a fixed random seed, and the common chronological holdout and preprocessing. It is evaluated with MAE, RMSE, and R². The fitted pipeline is saved as `models/random_forest.joblib`.
