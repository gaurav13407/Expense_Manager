# Data preprocessing guide

This document describes the preprocessing currently used for the Online Retail II workbook. It is intended as a plain-language reference; the implementation is in `scripts/preprocessing/prepare_dataset.py`.

## Source data

The input workbook is `Data/online_retail_II.xlsx`. It has two worksheets:

- `Year 2009-2010`
- `Year 2010-2011`

Each row represents one product line on an invoice. The workbook contains invoice and product identifiers, product description, quantity, invoice date, unit price, customer ID, and country.

## What the current preprocessing does

1. **Reads the fields needed for preparation.** It loads both sheets and keeps the invoice, stock code, description, quantity, invoice date, price, customer ID, and country columns.
2. **Combines both years.** The two worksheets are appended into one table so the records can be processed together.
3. **Standardizes names and types.** Column names are converted to lowercase with underscores; the invoice date is parsed as a date/time, quantity and price are converted to numbers, and product codes and country names are trimmed. Product codes are uppercased for consistent matching.
4. **Removes rows without required values.** A row is removed if it has no invoice date, product code, country, quantity, or price. A missing customer ID or description does not, by itself, remove a row.
5. **Excludes returns/cancellations from this model dataset.** Rows with zero or negative quantity are removed. These records are intended for a separate returns and cancellation analysis later.
6. **Excludes invalid prices.** Rows with zero or negative prices are removed because they do not describe a positively priced sales line for the current modeling task.
7. **Removes exact duplicate rows.** If every field in two rows is identical, one copy is removed.
8. **Adds line revenue.** `line_revenue` is calculated as `quantity × price`. This is a descriptive field in the prepared file; the current model target is quantity.
9. **Sorts records by date.** The output is ordered from oldest to newest, which makes chronological model evaluation possible.
10. **Writes a summary and step-by-step audit.** The summary records the source and prepared row counts, date range, unique products and countries, and customers with IDs. The audit records the number removed at each sequential cleaning step and the number remaining after that step.

## Current dataset counts

These counts were produced from the workbook during the latest preprocessing run:

| Measure | Result |
| --- | ---: |
| Rows in source sheets | 1,067,371 |
| Rows in prepared dataset | 1,007,914 |
| Rows removed or excluded | 59,457 |
| Date range | 2009-12-01 to 2011-12-09 |
| Unique product codes | 4,745 |
| Unique countries | 43 |
| Customers with an ID | 5,878 |

## Files produced

- `data/processed/online_retail_clean.csv` — cleaned invoice-line data, including the calculated `line_revenue` field.
- `data/processed/data_summary.csv` — source/prepared row counts and basic coverage details.
- `data/processed/preprocessing_audit.csv` — counts removed for missing required fields, blank product codes, non-positive quantities, non-positive prices, and exact duplicates.

## How the training script uses the prepared data

The current model predicts transaction-line quantity. It uses product code, country, price, and calendar values derived from the invoice date. Invoice number and customer ID are kept in the prepared file for reference but are not model inputs. The training script separately caps quantity at 1,000 for model fitting and price at 500 when building model features; those caps are model-preparation choices and do not alter the cleaned CSV.

## Current limits and possible next steps

- Returns and cancellations are excluded from this quantity model. They can be examined in a separate analysis, including cancellation markers, negative quantities, and returned value.
- The summary reports the total number of removed rows but does not yet break that count down by reason. A future version could report missing values, non-positive quantities, invalid prices, and duplicates separately.
- Unusually high quantities and prices are not removed from the prepared CSV. Their distributions should be reviewed before choosing a rule to flag, cap, or exclude them.
- Exact duplicate rows are removed without checking whether repeated-looking invoice lines may be legitimate. That assumption can be revisited if the analysis requires preserving each source line exactly.
- The prepared file preserves description and identifiers for traceability, while the current model uses only a subset of fields.

To recreate these files, run `python scripts/preprocessing/prepare_dataset.py` from the project root.
