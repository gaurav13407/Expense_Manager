"""Prepare Online Retail II invoice lines for the quantity-prediction project."""
from pathlib import Path

import pandas as pd

# This file is one folder below the project root.
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data" / "online_retail_II.xlsx"
OUT = ROOT / "data" / "processed"
REQUIRED = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
MODEL_REQUIRED = ["invoice_date", "stockcode", "country", "quantity", "price"]


def load_workbook(path: Path) -> pd.DataFrame:
    """Read the selected columns from every workbook sheet into one table."""
    sheets = pd.read_excel(path, sheet_name=None, usecols=REQUIRED, engine="openpyxl")
    if not sheets:
        raise ValueError(f"The workbook contains no readable sheets: {path}")
    # Concatenate worksheets in workbook order; retain their shared column layout.
    return pd.concat(sheets.values(), ignore_index=True)


def standardize_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, dates, numeric values, product codes, and countries."""
    data = data.copy()
    data.columns = [str(column).strip().lower().replace(" ", "_") for column in data.columns]
    # Excel calls this field InvoiceDate without a space, so normalize it explicitly.
    data = data.rename(columns={"invoicedate": "invoice_date"})
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    # Strip stray whitespace and standardize product-code casing for matching.
    data["stockcode"] = data["stockcode"].astype("string").str.strip().str.upper()
    data["country"] = data["country"].astype("string").str.strip()
    return data


def clean_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove unusable sales rows and return both clean rows and a step audit."""
    audit = []

    def record_step(label: str, before: int, after: int, reason: str) -> None:
        """Append row counts for one sequential cleaning rule to the audit."""
        audit.append({"step": label, "rows_before": before, "rows_removed": before - after,
                      "rows_after": after, "reason": reason})

    # Missing values in required fields make a line unusable for this prediction task.
    before = len(data)
    clean = data.dropna(subset=MODEL_REQUIRED).copy()
    record_step("missing_required_fields", before, len(clean),
                "Removed rows missing invoice date, product code, country, quantity, or price.")

    # Product codes converted to strings can be blank even when they are not null.
    before = len(clean)
    clean = clean[clean["stockcode"].str.len().fillna(0) > 0].copy()
    record_step("blank_product_code", before, len(clean), "Removed rows with an empty product code.")

    # Keep returns and cancellations out of this sales-quantity dataset for separate analysis.
    before = len(clean)
    clean = clean[clean["quantity"] > 0].copy()
    record_step("non_positive_quantity", before, len(clean),
                "Excluded zero/negative quantity rows, including returns and cancellations.")

    # Non-positive prices do not represent a positively priced sales line.
    before = len(clean)
    clean = clean[clean["price"] > 0].copy()
    record_step("non_positive_price", before, len(clean), "Removed rows with zero or negative unit price.")

    # Only exact repeated rows are dropped; records that differ in any field are retained.
    before = len(clean)
    clean = clean.drop_duplicates().copy()
    record_step("exact_duplicates", before, len(clean), "Removed exact duplicate rows across all source fields.")

    # Add revenue as a convenient descriptive measure; it is not the current model target.
    clean["line_revenue"] = clean["quantity"] * clean["price"]
    # Sort by time to support downstream chronological train/holdout splitting.
    clean = clean.sort_values("invoice_date").reset_index(drop=True)
    return clean, pd.DataFrame(audit)


def build_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Summarize source-to-clean row counts and coverage in the prepared dataset."""
    return pd.DataFrame([{
        "source_rows": len(raw),
        "prepared_rows": len(clean),
        "rows_removed": len(raw) - len(clean),
        "date_start": clean["invoice_date"].min(),
        "date_end": clean["invoice_date"].max(),
        "products": clean["stockcode"].nunique(),
        "countries": clean["country"].nunique(),
        "customers_with_id": clean["customer_id"].nunique(),
    }])


def main() -> None:
    """Run the complete preparation workflow and save clean data plus audit files."""
    # Verify the input exists and create the output folder before processing.
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)

    # Load, normalize, clean, and summarize in separate steps for easier review.
    raw = load_workbook(SOURCE)
    normalized = standardize_fields(raw)
    clean, audit = clean_transactions(normalized)
    summary = build_summary(raw, clean)

    # Write three reviewable outputs: cleaned records, overall counts, and per-rule audit.
    clean_path = OUT / "online_retail_clean.csv"
    clean.to_csv(clean_path, index=False)
    summary.to_csv(OUT / "data_summary.csv", index=False)
    audit.to_csv(OUT / "preprocessing_audit.csv", index=False)
    print(f"Prepared {len(clean):,} rows -> {clean_path}")
    print(summary.to_string(index=False))
    print("\nRows removed at each preprocessing step:")
    print(audit[["step", "rows_removed", "rows_after"]].to_string(index=False))


if __name__ == "__main__":
    main()
