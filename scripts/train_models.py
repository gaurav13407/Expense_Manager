"""Evaluate four regressors chronologically, then refit and save them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import load_category_spending, save_processed_data

from src.forecasting import (
    MODEL_DIR,
    OUTPUT_DIR,
    add_lag_features,
    build_models,
    chronological_evaluation,
    forecast_next_periods,
    save_models,
)


def main() -> None:
    data = load_category_spending()
    save_processed_data(data)
    metrics, holdout_predictions, _ = chronological_evaluation(data, holdout_periods=4)

    full_features = add_lag_features(data)
    x = full_features[["category", "period_index", "lag_1", "lag_2", "lag_3_mean"]]
    y = full_features["expenditure_gbp"]
    final_models = build_models()
    for model in final_models.values():
        model.fit(x, y)
    save_models(final_models)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    holdout_predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    forecasts = []
    for model_name, model in final_models.items():
        result = forecast_next_periods(data, model, periods=3)
        result.insert(0, "model", model_name)
        forecasts.append(result)
    pd.concat(forecasts, ignore_index=True).to_csv(OUTPUT_DIR / "three_period_forecasts.csv", index=False)

    print("Rolling four-period holdout comparison (lower MAE/RMSE is better):")
    print(metrics.to_string(index=False, formatters={"mae_gbp": "{:.2f}".format, "rmse_gbp": "{:.2f}".format, "r2": "{:.3f}".format}))
    print(f"Saved four fitted models in: {MODEL_DIR}")
    print(f"Saved evaluation and forecasts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
