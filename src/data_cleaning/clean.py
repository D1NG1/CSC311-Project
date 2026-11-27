"""Clean the raw survey data with fixed rules (nothing is learned from data). Uses only numpy/pandas."""
import numpy as np
import pandas as pd

from data_cleaning.config import (
    COLUMNS, LIKERT_MAP,
    MISSING_PATTERN, ZERO_PRICE_PATTERN, NUMBER_PATTERN, RANGE_PATTERN,
    THOUSANDS_SPACE_PATTERN, NOT_PRICE_PATTERN, RATING_COL, LIKERT_COLS, MULTI_COLS, TEXT_COLS, PRICE_COL,
)


def rename_columns(df):
    """Replace the long survey questions with short column names."""
    return df.rename(columns=COLUMNS)


def normalize_missing(s):
    """Strip whitespace and turn empty or "no answer" text into NaN."""
    s = s.astype("string").str.strip()
    is_missing = s.isna() | (s == "") | s.str.fullmatch(MISSING_PATTERN).fillna(False)
    return s.mask(is_missing, np.nan)


def parse_text_col(s):
    """Clean a free-text column: "no answer" -> NaN, lowercase, remove punctuation, collapse whitespace."""
    s = normalize_missing(s).str.lower()
    s = s.str.replace(r"['’]", "", regex=True)  # "don't" -> "dont", not "don t"
    s = s.str.replace(r"[^\w\s]", " ", regex=True)  # other punctuation -> space
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s == "", np.nan)


def parse_price(text):
    """Turn one price answer into a number, e.g. "$1,000" -> 1000. NaN if unparseable."""
    if pd.isna(text):
        return np.nan
    text = THOUSANDS_SPACE_PATTERN.sub("", str(text).replace(",", ""))
    if NOT_PRICE_PATTERN.search(text):  # answered with a rating, not a price
        return np.nan
    numbers = [float(n) * (1000 if k else 1) for n, k in NUMBER_PATTERN.findall(text)]
    if numbers:
        # A range like "100-200" -> 150, otherwise take the first number
        if RANGE_PATTERN.search(text) and len(numbers) >= 2:
            return (numbers[0] + numbers[1]) / 2
        return numbers[0]
    if ZERO_PRICE_PATTERN.search(text):
        return 0.0
    return np.nan


def parse_price_col(s):
    """Apply parse_price to a whole column."""
    return s.map(parse_price).astype(float)


def parse_likert_col(s):
    """Map a Likert column to 1-5, e.g. "Agree" -> 4. Unknown or empty -> NaN."""
    return s.astype("string").str.strip().map(LIKERT_MAP).astype(float)


def parse_rating_col(s):
    """Convert the 0-10 rating to numbers. Non-numeric or out-of-range -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(0, 10)).astype(float)


def clean(df):
    """
    Apply all fixed cleaning rules to a raw survey table. Column names become short
    and values become clean but still readable (multi-select stays as "A,B,C").
    """
    df = rename_columns(df).copy()
    df[RATING_COL] = parse_rating_col(df[RATING_COL])
    for col in LIKERT_COLS:
        df[col] = parse_likert_col(df[col])
    for col in MULTI_COLS:
        df[col] = df[col].astype("string").str.strip()  # "No particular type" is a real option
    for col in TEXT_COLS:
        df[col] = parse_text_col(df[col])
    df[PRICE_COL] = parse_price_col(df[PRICE_COL])
    return df


def drop_empty_rows(df):
    """Drop rows where every feature is missing. Use on training data only."""
    features = [RATING_COL, *LIKERT_COLS, *MULTI_COLS, *TEXT_COLS, PRICE_COL]
    return df[df[features].notna().any(axis=1)]


if __name__ == "__main__":
    # Self-check: uv run python -m data_cleaning.clean  (from src/)
    cases = {
        "$1,000": 1000, "5k": 5000, "$10 000": 10000, "100CAD": 100,
        "400 - 700": 550, "CAD $80–$120": 100, "50 to 150 CAD": 100,
        "Not willing.": 0, "Zero": 0, "Thirty dollars": None, "n/a": None,
        "1 to 10 scale i will put a 2": None,
    }
    for text, want in cases.items():
        got = parse_price(text)
        assert (np.isnan(got) if want is None else got == want), (text, got, want)
    s = parse_text_col(pd.Series(["Don’t know", "nil", "I don't like it!", "  Blue. "]))
    assert s.isna().tolist() == [True, True, False, False], s.tolist()
    assert s[2:].tolist() == ["i dont like it", "blue"], s.tolist()
    print("clean.py self-check passed")
