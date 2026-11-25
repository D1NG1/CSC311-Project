"""Shared 5-fold grouped cross-validation utilities for the training split."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from data_cleaning.clean import clean, drop_empty_rows
from data_preprocessing.features import Featurizer

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "splits" / "train.csv"
RESULTS_DIR = ROOT / "results"
CLASSES = ["Baseball", "Basketball", "Soccer"]
N_SPLITS = 5
SEED = 42


def load_train_df():
    """Load and clean the training split, dropping completely empty rows."""
    df = pd.read_csv(TRAIN_FILE)
    return drop_empty_rows(clean(df)).reset_index(drop=True)


def get_folds(df, n_splits=N_SPLITS, seed=SEED):
    """Build stratified cross-validation folds grouped by student id."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, df["label"], df["id"]))


def feature_grid():
    """Return all feature encoding combinations to tune."""
    combos = itertools.product(
        ["standardize", "divide", "raw"],
        ["ordinal", "onehot"],
        [True, False],
        [True, False],
    )

    return [
        {
            "rating": rating,
            "likert": likert,
            "log_price": log_price,
            "use_strategy": use_strategy,
        }
        for rating, likert, log_price, use_strategy in combos
    ]


def featurize_folds(df, folds, feature_config):
    """Fit Featurizer on each training fold and return (X_train, y_train, X_val, y_val) splits."""
    splits = []

    for train_idx, val_idx in folds:
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        featurizer = Featurizer(**feature_config)
        X_train = featurizer.fit_transform(train_df)
        X_val = featurizer.transform(val_df)

        y_train = train_df["label"].to_numpy()
        y_val = val_df["label"].to_numpy()

        splits.append((X_train, y_train, X_val, y_val))

    return splits


def evaluate_cv(fold_splits, make_model):
    """Fit and evaluate make_model() across all CV folds."""
    train_accs = []
    val_accs = []
    val_f1s = []

    for X_train, y_train, X_val, y_val in fold_splits:
        model = make_model()
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_accs.append(accuracy_score(y_train, pred_train))
        val_accs.append(accuracy_score(y_val, pred_val))
        val_f1s.append(f1_score(y_val, pred_val, average="macro", labels=CLASSES, zero_division=0))

    return {
        "val_acc": float(np.mean(val_accs)),
        "val_acc_std": float(np.std(val_accs)),
        "val_f1": float(np.mean(val_f1s)),
        "train_acc": float(np.mean(train_accs)),
    }


def score_key(row):
    """Rank configurations by validation accuracy, breaking ties with macro-F1."""
    return (row["val_acc"], row["val_f1"])


def evaluate_grid(df, folds, feature_configs, param_grid, make_model):
    """Evaluate every (feature_config, hyperparams) pair across CV folds."""
    rows = []

    for feature_config in feature_configs:
        fold_splits = featurize_folds(df, folds, feature_config)

        for params in param_grid:
            metrics = evaluate_cv(fold_splits, lambda: make_model(**params))
            rows.append({**feature_config, **params, **metrics})

    return rows


def save_results(name, rows):
    """Write CV result rows to results/{name}.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def print_top_results(rows, top_k=10):
    """Print top_k CV rows sorted by validation accuracy then macro-F1."""
    ranked = sorted(rows, key=score_key, reverse=True)[:top_k]
    print(pd.DataFrame(ranked).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
