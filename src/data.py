"""Load and prepare the ONS household spending time series."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "data" / "raw" / "workbook1detailedexpenditureandtrends.xlsx"
PROCESSED_CSV = ROOT / "data" / "processed" / "category_spending.csv"


def _clean_period(value: object, index: int) -> str:
    """Remove Excel footnote suffixes while keeping repeated 2006 periods distinct."""
    label = str(value).strip()
    match = re.fullmatch(r"(\d{4}-\d{2})\d*", label)
    if match:
        return match.group(1)
    if label in {"20062", "20063"}:
        return f"2006 (series {label[-1]})"
    return label


def load_category_spending(workbook_path: str | Path = DEFAULT_WORKBOOK) -> pd.DataFrame:
    """Convert source sheet 4.1 from a wide table into one row per category-period."""
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Source workbook not found: {workbook_path}")

    raw = pd.read_excel(workbook_path, sheet_name="4.1", header=None, engine="openpyxl")
    first_year_col = 4  # Excel column E; pandas is zero based.
    labels = raw.iloc[6, first_year_col:].tolist()
    periods = [(_clean_period(label, i), label) for i, label in enumerate(labels) if pd.notna(label)]
    rows: list[dict[str, object]] = []

    for row_index, row in raw.iterrows():
        code = row.iloc[0]
        category = row.iloc[1]
        if pd.isna(code) or pd.isna(category):
            continue
        try:
            code_num = int(code)
        except (TypeError, ValueError):
            continue
        if code_num < 1 or code_num > 12 or float(code) != code_num:
            continue

        for period_index, (period_label, original_label) in enumerate(periods):
            value = row.iloc[first_year_col + period_index]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                value = value.replace(",", "").replace("£", "").strip()
                value = re.sub(r"[~\[\]]", "", value)
                try:
                    value = float(value)
                except ValueError:
                    continue
            rows.append(
                {
                    "category_code": code_num,
                    "category": re.sub(r"(?<=\w)\d+(?=,)", "", str(category).strip()).rstrip("1234567890").strip(),
                    "period": period_label,
                    "source_period": str(original_label),
                    "period_index": period_index,
                    "expenditure_gbp": float(value),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["category_code", "period_index"]).reset_index(drop=True)
    if frame.empty or frame["category"].nunique() != 12:
        raise ValueError("Could not find all 12 category series in worksheet 4.1.")
    return frame


def save_processed_data(frame: pd.DataFrame, destination: str | Path = PROCESSED_CSV) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        frame = load_category_spending()
        save_processed_data(frame)
        return frame
    frame = pd.read_csv(PROCESSED_CSV)
    required = {"category", "period", "period_index", "expenditure_gbp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Processed spending data is missing columns: {sorted(missing)}")
    return frame


def load_income_decile_spending(workbook_path: str | Path = DEFAULT_WORKBOOK) -> pd.DataFrame:
    """Read the current-year gross-income decile comparison from source table A6."""
    raw = pd.read_excel(workbook_path, sheet_name="A6", header=None, engine="openpyxl")
    deciles = ["Lowest", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Highest"]
    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        code, category = row.iloc[0], row.iloc[1]
        if pd.isna(code) or pd.isna(category):
            continue
        try:
            code_num = int(code)
        except (TypeError, ValueError):
            continue
        if code_num < 1 or code_num > 12 or float(code) != code_num:
            continue
        for offset, decile in enumerate(deciles):
            value = row.iloc[2 + offset]
            if pd.notna(value):
                try:
                    rows.append({"category": str(category).strip(), "income_decile": decile, "expenditure_gbp": float(value)})
                except (TypeError, ValueError):
                    pass
    return pd.DataFrame(rows)
