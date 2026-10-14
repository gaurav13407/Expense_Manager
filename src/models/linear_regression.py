"""Member 1: linear trend and lag-based regression baseline."""

from sklearn.linear_model import LinearRegression


def create_estimator() -> LinearRegression:
    return LinearRegression()
