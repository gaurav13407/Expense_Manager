# Extra Trees Regressor

## What we built

The Extra Trees script is `scripts/training/train_extra_trees.py`. It trains an ensemble of randomized decision trees and averages their outputs to estimate transaction-line quantity.

## Why we included it

Extra Trees is closely related to Random Forest, but it adds more randomness when choosing split points. Comparing the two helps us see whether this stronger randomization improves generalization on this retail dataset.

## Advantages

- Models nonlinear patterns and combinations of features.
- More randomized splits can reduce similarity between trees and help control overfitting.
- Often trains efficiently because its split choices are randomized.
- Offers a useful direct comparison with Random Forest under the same inputs and data split.

## Trade-offs

- Randomized splits can miss useful boundaries, so it may perform worse than Random Forest on some data.
- Like other tree ensembles, it can be large and does not naturally extrapolate beyond observed patterns.
- The shared integer encoding of categorical values imposes arbitrary order in tree splits; this is a possible improvement area.

## Shared training choices

The script uses 50 trees, a fixed random seed, and the common chronological holdout and preprocessing. It is evaluated with MAE, RMSE, and R². The fitted pipeline is saved as `models/extra_trees.joblib`.
