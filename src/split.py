"""Split the data into train/test sets by student id."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "csc311_project_training_202609.csv"
SPLIT_DIR = ROOT / "data" / "splits"

TEST_FRAC = 0.2
SEED = 42


def split_by_student(ids, test_frac=TEST_FRAC, seed=SEED):
    """Keep all rows of a student in the same split."""
    rng = np.random.default_rng(seed)
    students = rng.permutation(ids.unique())
    n_test = int(len(students) * test_frac)
    n_train = len(students) - n_test
    test_ids = students[:n_test]
    train_ids = students[n_test:n_test + n_train]
    return ids.isin(train_ids), ids.isin(test_ids)


def main():
    # 1. Load data
    df = pd.read_csv(DATA_FILE)

    # 2. Split Data by student (80% Train, 20% Test)
    train, test = split_by_student(df["id"])
    df_train, df_test = df[train], df[test]
    print(f"Data Splits: Train={len(df_train)}, Test={len(df_test)}")

    # 3. Save Splits
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(SPLIT_DIR / "train.csv", index=False)
    df_test.to_csv(SPLIT_DIR / "test.csv", index=False)

if __name__ == "__main__":
    main()
