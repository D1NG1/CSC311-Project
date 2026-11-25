"""Shared data loading and grid-search utilities for the training scripts."""
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from data_cleaning.clean import clean, drop_empty_rows
from data_preprocessing.features import Featurizer

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "splits" / "train.csv"
RESULTS_DIR = ROOT / "results"
N_SPLITS = 5
SEED = 42

FEATURE_GRID = {
    "features__rating": ["standardize", "divide", "raw"],
    "features__likert": ["ordinal", "onehot"],
    "features__log_price": [True, False],
    "features__use_strategy": [True, False],
}


class SklearnFeaturizer(TransformerMixin, BaseEstimator, Featurizer):
    """
    Featurizer class modified to match sklearn's conventions.
    For use in sklearn Pipeline and GridSearchCV.
    """

    def fit(self, X, y=None):
        return super().fit(X)


def load_train_df():
    """Load and clean the training split, dropping completely empty rows."""
    df = pd.read_csv(TRAIN_FILE)
    return drop_empty_rows(clean(df)).reset_index(drop=True)


def get_folds(df, n_splits=N_SPLITS, seed=SEED):
    """Build stratified cross-validation folds grouped by student id."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, df["label"], df["id"]))


def grid_search(df, model: BaseEstimator, param_grid: dict[str, list]):
    """Cross-validate a Featurizer + model pipeline over param_grid, best configs first."""
    pipeline = Pipeline([("features", SklearnFeaturizer()), ("model", model)])
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring={"acc": "accuracy", "f1": "f1_macro"},
        cv=get_folds(df),
        refit=False,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(df, df["label"])

    return results_table(search.cv_results_)


def results_table(cv_results):
    """Turn GridSearchCV results into one row per config, sorted by val accuracy then macro-F1."""
    params = pd.DataFrame(cv_results["params"]).rename(columns=lambda name: name.split("__")[-1])
    scores = pd.DataFrame({
        "val_acc": cv_results["mean_test_acc"],
        "val_acc_std": cv_results["std_test_acc"],
        "val_f1": cv_results["mean_test_f1"],
        "train_acc": cv_results["mean_train_acc"],
    })

    results = pd.concat([params, scores], axis=1)
    return results.sort_values(["val_acc", "val_f1"], ascending=False, ignore_index=True)


def save_results(name, results):
    """Write a results table to results/{name}.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{name}.csv"
    results.to_csv(csv_path, index=False)
    return csv_path


def print_top_results(results, top_k=10):
    """Print the top_k rows of a sorted results table."""
    print(results.head(top_k).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
