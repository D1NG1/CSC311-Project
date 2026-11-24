"""Turn cleaned survey data into numeric features. Uses only numpy/pandas."""
import numpy as np
import pandas as pd

from data_cleaning.config import (
    MULTI_OPTIONS, RATING_COL, LIKERT_COLS, MULTI_COLS, TEXT_COLS, PRICE_COL,
)
from data_preprocessing.text import TextVectorizer


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


class Featurizer:
    """
    Cleaned survey table (output of clean()) -> numeric matrix X.
    fit() learns medians, scaling and TF-IDF vocabularies from the training split only;
    transform() applies them to any split. Column names are in self.names after transform().

    Options, compared by cross-validation:
      rating:       "standardize" | "divide" (x / 10) | "raw"
      likert:       "ordinal" (one 1-5 column per question) | "onehot" (5 columns per question)
      log_price:    apply log(1 + x) to price before standardizing it
      use_strategy: keep the strategy Likert question
      min_count:    TF-IDF keeps words found in at least this many training answers
    """

    def __init__(self, rating="standardize", likert="ordinal", log_price=True,
                 use_strategy=True, min_count=3):
        self.rating = rating
        self.likert = likert
        self.log_price = log_price
        self.use_strategy = use_strategy
        self.min_count = min_count

    def _price(self, df):
        p = df[PRICE_COL].fillna(self.medians[PRICE_COL]).to_numpy(float)
        return np.log1p(p) if self.log_price else p

    def fit(self, df):
        self.likert_cols = [c for c in LIKERT_COLS if self.use_strategy or c != "strategy"]

        # Medians for imputing missing values. Likert medians are rounded so one-hot still works.
        self.medians = df[[RATING_COL, *self.likert_cols, PRICE_COL]].median()
        self.medians[self.likert_cols] = self.medians[self.likert_cols].round()

        rating = df[RATING_COL].fillna(self.medians[RATING_COL])
        self.rating_mean, self.rating_std = rating.mean(), rating.std()
        price = self._price(df)
        self.price_mean, self.price_std = price.mean(), price.std()

        self.text = {c: TextVectorizer(self.min_count).fit(df[c]) for c in TEXT_COLS}
        return self

    def transform(self, df):
        df = df.fillna(self.medians)  # fills only the rating, Likert and price columns
        blocks, names = [], []

        def add(values, cols):
            blocks.append(np.asarray(values, dtype=float).reshape(len(df), -1))
            names.extend(cols)

        # Rating (0-10)
        rating = df[RATING_COL].to_numpy(float)
        if self.rating == "standardize":
            rating = (rating - self.rating_mean) / self.rating_std
        elif self.rating == "divide":
            rating = rating / 10
        add(rating, [RATING_COL])

        # Likert (1-5)
        for c in self.likert_cols:
            v = df[c].to_numpy(float)
            if self.likert == "onehot":
                add(v[:, None] == np.arange(1, 6), [f"{c}_{k}" for k in range(1, 6)])
            else:
                add(v, [c])

        # Multi-select: one 0/1 column per option
        for c in MULTI_COLS:
            m = parse_multi_col(df[c], c)
            add(m, list(m.columns))

        # Free text: TF-IDF per question, plus a 0/1 "no answer" column
        for c in TEXT_COLS:
            vec = self.text[c]
            add(vec.transform(df[c].tolist()), [f"{c}:{w}" for w in vec.vocab])
            add(df[c].isna(), [f"{c}_missing"])

        # Price: impute, optional log, then standardize
        add((self._price(df) - self.price_mean) / self.price_std, [PRICE_COL])

        self.names = names
        return np.hstack(blocks)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


if __name__ == "__main__":
    # Self-check: uv run python -m data_preprocessing.features  (from src/)
    from pathlib import Path
    from data_cleaning.clean import clean, drop_empty_rows

    train = drop_empty_rows(clean(pd.read_csv(Path(__file__).parents[2] / "data/splits/train.csv")))
    empty = train.iloc[:1].copy()
    empty[[c for c in empty.columns if c not in ("id", "label")]] = np.nan  # an all-missing row

    for opts in [{}, {"likert": "onehot", "rating": "divide", "log_price": False, "use_strategy": False}]:
        f = Featurizer(**opts)
        X = f.fit_transform(train)
        assert X.shape == (len(train), len(f.names)), (X.shape, len(f.names))
        assert np.isfinite(X).all()
        X_empty = f.transform(empty)  # test-time rows may be empty: must still produce a row
        assert X_empty.shape == (1, X.shape[1]) and np.isfinite(X_empty).all()
        print(opts or "defaults", "->", X.shape)

    f = Featurizer(likert="onehot").fit(train)
    X = f.transform(train)
    onehot = [i for i, n in enumerate(f.names) if n[:-2] in LIKERT_COLS and n[-2] == "_"]
    assert (X[:, onehot].sum(axis=1) == 3).all()  # exactly one answer per Likert question
    print("features.py self-check passed")
