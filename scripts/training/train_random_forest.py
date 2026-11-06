"""Train and evaluate the Random Forest quantity-prediction model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "online_retail_clean.csv"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "outputs"
SEED, MAX_TRAIN, MAX_TEST = 42, 60_000, 20_000
MODEL_NAME = "Random Forest"
MODEL_FILE = "random_forest.joblib"
PREDICTIONS_FILE = "random_forest_predictions.csv"


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare product, country, price, and calendar inputs for Random Forest."""
    dates = pd.to_datetime(frame["invoice_date"])
    return pd.DataFrame({
        "stockcode": frame["stockcode"].astype(str),
        "country": frame["country"].astype(str),
        "price": frame["price"].astype(float).clip(upper=500),
        "year": dates.dt.year, "month": dates.dt.month, "day": dates.dt.day,
        "weekday": dates.dt.weekday, "hour": dates.dt.hour,
    })


def load_split_data():
    """Load prepared data and reserve the final two months for evaluation."""
    if not INPUT.exists():
        raise FileNotFoundError(f"Run scripts/preprocessing/prepare_dataset.py first; missing {INPUT}")
    data = pd.read_csv(INPUT, parse_dates=["invoice_date"], low_memory=False)
    cutoff = data["invoice_date"].max().to_period("M") - 1
    train = data[data["invoice_date"].dt.to_period("M") < cutoff].copy()
    test = data[data["invoice_date"].dt.to_period("M") >= cutoff].copy()
    if len(train) < 100 or len(test) < 100:
        raise ValueError("Not enough records on both sides of the chronological split.")
    if len(train) > MAX_TRAIN:
        train = train.sample(MAX_TRAIN, random_state=SEED)
    if len(test) > MAX_TEST:
        test = test.sample(MAX_TEST, random_state=SEED)
    X_train, X_test = create_features(train), create_features(test)
    y_train = np.log1p(train["quantity"].clip(upper=1000).to_numpy())
    y_test = test["quantity"].clip(upper=1000).to_numpy()
    return train, test, X_train, X_test, y_train, y_test


def create_preprocessor() -> ColumnTransformer:
    """Encode categories and impute/scale numbers as part of the forest pipeline."""
    categorical = ["stockcode", "country"]
    numeric = ["price", "year", "month", "day", "weekday", "hour"]
    return ColumnTransformer([
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ])


def main() -> None:
    """Fit Random Forest, score the holdout, and save its pipeline and predictions."""
    train, test, X_train, X_test, y_train, y_test = load_split_data()
    estimator = RandomForestRegressor(
        n_estimators=50, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=SEED,
    )
    model = Pipeline([("preprocess", create_preprocessor()), ("model", estimator)])
    model.fit(X_train, y_train)
    predicted = np.maximum(0, np.expm1(model.predict(X_test)))
    metrics = {
        "model": MODEL_NAME,
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": mean_squared_error(y_test, predicted) ** 0.5,
        "R2": r2_score(y_test, predicted),
        "train_rows": len(train), "test_rows": len(test),
        "test_start": test["invoice_date"].min(), "test_end": test["invoice_date"].max(),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / MODEL_FILE)
    predictions = test[["invoice", "invoice_date", "stockcode", "country", "quantity", "price"]].copy()
    predictions["actual_quantity_capped"] = y_test
    predictions["predicted_quantity"] = predicted
    predictions.to_csv(OUT / PREDICTIONS_FILE, index=False)
    comparison_path = OUT / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    if not comparison.empty:
        comparison = comparison[comparison["model"] != MODEL_NAME]
    pd.concat([comparison, pd.DataFrame([metrics])], ignore_index=True).sort_values("MAE").to_csv(comparison_path, index=False)
    print(f"{MODEL_NAME}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")
    print(f"Saved {MODEL_DIR / MODEL_FILE} and {OUT / PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
