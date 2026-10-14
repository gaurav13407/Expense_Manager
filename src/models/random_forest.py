"""Member 2: random forest regression for nonlinear spending patterns."""

from sklearn.ensemble import RandomForestRegressor


def create_estimator() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
