"""Member 3: gradient boosting regression with robust Huber loss."""

from sklearn.ensemble import GradientBoostingRegressor


def create_estimator() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=2,
        loss="huber",
        random_state=42,
    )
