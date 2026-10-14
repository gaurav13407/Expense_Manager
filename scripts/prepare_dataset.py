"""Create the tidy category-by-period table used by all four models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_category_spending, save_processed_data


if __name__ == "__main__":
    frame = load_category_spending()
    destination = save_processed_data(frame)
    print(f"Prepared {len(frame)} category-period observations across {frame['category'].nunique()} categories.")
    print(f"Saved: {destination}")
