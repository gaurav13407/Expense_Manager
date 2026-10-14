"""Member 4: support vector regression for a smooth nonlinear fit."""

from sklearn.svm import SVR


def create_estimator() -> SVR:
    return SVR(kernel="rbf", C=20, epsilon=0.15, gamma="scale")
