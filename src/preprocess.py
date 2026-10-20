"""Clean the survey data and build numeric features. Uses only numpy/pandas."""
import numpy as np
import pandas as pd

from config import (
    COLUMNS, LIKERT_MAP, MULTI_OPTIONS,
    MISSING_PATTERN, ZERO_PRICE_PATTERN, NUMBER_PATTERN, RANGE_PATTERN,
    RATING_COL, LIKERT_COLS, MULTI_COLS, TEXT_COLS, PRICE_COL,
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
    s = s.str.replace(r"[^\w\s]", " ", regex=True)  # punctuation -> space
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s == "", np.nan)


def parse_price(text):
    """Turn one price answer into a number, e.g. "$1,000" -> 1000. NaN if unparseable."""
    if pd.isna(text):
        return np.nan
    text = str(text).replace(",", "")
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


def parse_multi_col(s, col):
    """
    Turn a multi-select column into one 0/1 column per option,
    e.g. "Spring,Summer" -> season_Winter=0, season_Spring=1, season_Summer=1, season_Fall=0.
    Empty answers get all zeros.
    """
    chosen = s.astype("string").fillna("").str.split(",").map(lambda xs: {x.strip() for x in xs})
    return pd.DataFrame(
        {f"{col}_{opt}": chosen.map(lambda xs: int(opt in xs)) for opt in MULTI_OPTIONS[col]},
        index=s.index,
    )


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
