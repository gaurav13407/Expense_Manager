"""Four comparable regressors for category-level household expenditure forecasts."""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models import gradient_boosting, linear_regression, random_forest, support_vector_regression

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
NUMERIC_FEATURES = ["period_index", "lag_1", "lag_2", "lag_3_mean"]
CATEGORY_FEATURE = ["category"]


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.sort_values(["category", "period_index"]).copy()
    grouped = result.groupby("category", sort=False)["expenditure_gbp"]
    result["lag_1"] = grouped.shift(1)
    result["lag_2"] = grouped.shift(2)
    result["lag_3_mean"] = grouped.transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    return result.dropna(subset=["lag_1", "lag_2"]).reset_index(drop=True)


def _pipeline(estimator: object) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORY_FEATURE),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("features", prep), ("regressor", estimator)])


def build_models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": _pipeline(linear_regression.create_estimator()),
        "Random Forest": _pipeline(random_forest.create_estimator()),
        "Gradient Boosting": _pipeline(gradient_boosting.create_estimator()),
        "Support Vector Regression": _pipeline(support_vector_regression.create_estimator()),
    }


def chronological_evaluation(data: pd.DataFrame, holdout_periods: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    """Rolling one-step-ahead holdout; past observed values are available at each step."""
    features = add_lag_features(data)
    last_index = int(data["period_index"].max())
    first_test = last_index - holdout_periods + 1
    train = features[features["period_index"] < first_test]
    test = features[features["period_index"] >= first_test]
    x_train = train[CATEGORY_FEATURE + NUMERIC_FEATURES]
    y_train = train["expenditure_gbp"]
    x_test = test[CATEGORY_FEATURE + NUMERIC_FEATURES]
    y_test = test["expenditure_gbp"]

    metrics: list[dict[str, object]] = []
    predictions = test[["category", "period", "period_index", "expenditure_gbp"]].rename(columns={"expenditure_gbp": "actual_gbp"}).copy()
    fitted: dict[str, Pipeline] = {}
    for name, model in build_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        pred = np.maximum(model.predict(x_test), 0)
        predictions[name] = pred
        metrics.append({
            "model": name,
            "mae_gbp": mean_absolute_error(y_test, pred),
            "rmse_gbp": mean_squared_error(y_test, pred) ** 0.5,
            "r2": r2_score(y_test, pred),
            "training_rows": len(train),
            "holdout_rows": len(test),
            "holdout_periods": holdout_periods,
        })
        fitted[name] = model
    return pd.DataFrame(metrics).sort_values("mae_gbp").reset_index(drop=True), predictions, fitted


def forecast_next_periods(
    data: pd.DataFrame,
    model: Pipeline,
    periods: int = 3,
    starting_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Recursively forecast each category beyond its latest observed period."""
    if periods < 1:
        raise ValueError("Forecast horizon must be at least one period.")
    rows: list[dict[str, object]] = []
    for category, group in data.sort_values("period_index").groupby("category"):
        history = list(group["expenditure_gbp"].astype(float))
        if starting_values and category in starting_values:
            history[-1] = max(0.0, float(starting_values[category]))
        next_index = int(group["period_index"].max()) + 1
        for step in range(periods):
            x = pd.DataFrame([{
                "category": category,
                "period_index": next_index + step,
                "lag_1": history[-1],
                "lag_2": history[-2],
                "lag_3_mean": float(np.mean(history[-3:])),
            }])
            value = max(0.0, float(model.predict(x[CATEGORY_FEATURE + NUMERIC_FEATURES])[0]))
            rows.append({"category": category, "period_index": next_index + step, "forecast_gbp": value, "step": step + 1})
            history.append(value)
    return pd.DataFrame(rows)


def save_models(models: dict[str, Pipeline], directory: str | Path = MODEL_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_names = {
        "Linear Regression": "linear_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Support Vector Regression": "support_vector_regression.joblib",
    }
    for name, model in models.items():
        joblib.dump(model, directory / file_names[name])


def load_saved_model(name: str) -> Pipeline:
    paths = {
        "Linear Regression": "linear_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Support Vector Regression": "support_vector_regression.joblib",
    }
    return joblib.load(MODEL_DIR / paths[name])
