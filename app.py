"""Streamlit dashboard for UK household spending trends and forecasts."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_income_decile_spending, load_processed_data
from src.forecasting import MODEL_DIR, build_models, forecast_next_periods, load_saved_model

st.set_page_config(page_title="Household Spending Trends", page_icon="📊", layout="wide")
st.title("UK Household Spending Trends")
st.caption("Explore average weekly household expenditure by category and compare four forecasting algorithms.")

data = load_processed_data()
model_names = list(build_models())
years = data[["period_index", "period"]].drop_duplicates().sort_values("period_index")
year_labels = years["period"].tolist()

if not all((MODEL_DIR / filename).exists() for filename in ["linear_regression.joblib", "random_forest.joblib", "gradient_boosting.joblib", "support_vector_regression.joblib"]):
    st.warning("The saved models are not ready yet. From the project folder, run `python scripts/train_models.py`, then refresh this page.")
    st.stop()

comparison_path = Path(__file__).parent / "outputs" / "model_comparison.csv"
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path).sort_values("mae_gbp")
else:
    comparison = pd.DataFrame()

with st.sidebar:
    st.header("Explore")
    categories = sorted(data["category"].unique())
    selected_category = st.selectbox("Spending category", categories)
    selected_model = st.selectbox("Forecasting model", model_names)
    horizon = st.slider("Forecast horizon (years)", min_value=1, max_value=3, value=3)

category_data = data[data["category"] == selected_category].sort_values("period_index")
latest = float(category_data.iloc[-1]["expenditure_gbp"])
with st.sidebar:
    use_scenario = st.checkbox("Try a manual spending scenario")
    scenario_amount = st.number_input(
        "Recent weekly spend (£)",
        min_value=0.0,
        max_value=10000.0,
        value=latest,
        step=5.0,
        disabled=not use_scenario,
        help="Temporarily replace the latest observed category amount and see how the selected model's forecast responds.",
    )
    st.divider()
    st.caption("Source: UK Living Costs and Food Survey, table 4.1. Values are average weekly household spending at 2024–25 prices.")

model = load_saved_model(selected_model)
starting_values = {selected_category: scenario_amount} if use_scenario else None
forecasts = forecast_next_periods(data, model, periods=horizon, starting_values=starting_values)
category_forecast = forecasts[forecasts["category"] == selected_category].copy()
category_forecast["period"] = [f"Forecast {i + 1}" for i in range(len(category_forecast))]

next_value = float(category_forecast.iloc[0]["forecast_gbp"])
comparison_value = scenario_amount if use_scenario else latest
next_value = float(category_forecast.iloc[0]["forecast_gbp"])
change = next_value - comparison_value
change_pct = (change / comparison_value * 100) if comparison_value else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Latest observed period", str(category_data.iloc[-1]["period"]))
metric_cols[1].metric("Latest average weekly spend", f"£{latest:,.2f}")
metric_cols[2].metric("Next period estimate", f"£{next_value:,.2f}", f"{change_pct:+.1f}% vs {'scenario' if use_scenario else 'latest'}")
metric_cols[3].metric("Model selected", selected_model)

forecast_tab, compare_tab, income_tab, data_tab = st.tabs(["Forecast", "Compare models", "Income groups", "Prepared data"])

with forecast_tab:
    st.subheader(f"{selected_category}: historical spending and forecast")
    history_chart = category_data.set_index("period")["expenditure_gbp"].rename("Observed (£ per week)")
    future_chart = category_forecast.set_index("period")["forecast_gbp"].rename("Forecast (£ per week)")
    st.line_chart(pd.concat([history_chart, future_chart], axis=1), y_label="Pounds per week")
    if use_scenario:
        st.info(f"Scenario active: the latest observed amount (£{latest:,.2f}) is temporarily replaced with £{scenario_amount:,.2f} for this category's forecast. It does not change the source data.")
    st.dataframe(
        category_forecast[["period", "forecast_gbp"]].rename(columns={"period": "Forecast step", "forecast_gbp": "Estimated weekly spend (£)"}).style.format({"Estimated weekly spend (£)": "£{:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.info("Forecasts extend historical group averages. They are not predictions for an individual household. The model rolls its own forecasts forward after the first year.")

with compare_tab:
    st.subheader("Four-model evaluation")
    if comparison.empty:
        st.info("No evaluation file found. Run the training script to create model-comparison results.")
    else:
        st.caption("Evaluation uses the latest four source periods as a rolling one-step-ahead holdout. Lower MAE and RMSE indicate smaller errors.")
        st.dataframe(comparison.style.format({"mae_gbp": "£{:,.2f}", "rmse_gbp": "£{:,.2f}", "r2": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("model")[["mae_gbp", "rmse_gbp"]], y_label="Error (£ per week)")
        st.markdown("**How to read this:** MAE is the average absolute miss; RMSE penalizes larger misses more. R² is included as a secondary measure and can be negative on a small holdout.")

with income_tab:
    st.subheader("Spending by household income group")
    income = load_income_decile_spending()
    selected_income = income[income["category"] == selected_category]
    if selected_income.empty:
        st.info("No income-group comparison was available for this category.")
    else:
        st.bar_chart(selected_income.set_index("income_decile")["expenditure_gbp"], y_label="Average weekly household spend (£)")
        st.caption("Income deciles compare household averages in the latest survey year; they are contextual comparisons, not extra training rows.")

with data_tab:
    st.subheader("Model input data")
    st.write(f"{len(data):,} category-period observations covering {data['category'].nunique()} categories and {len(year_labels)} source periods.")
    st.dataframe(data, use_container_width=True, hide_index=True)

st.divider()
st.caption("For learning and demonstration. The workbook contains aggregated survey statistics, so results should not be used as personal financial advice.")
