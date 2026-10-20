"""Shared setup for the EDA scripts. Training split only."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # so we can import from src/

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from preprocess import clean, drop_empty_rows, rename_columns  # noqa: E402

TRAIN = SRC.parent / "data" / "splits" / "train.csv"
FIG_DIR = SRC.parent / "report" / "figures"
CLASSES = ["Baseball", "Basketball", "Soccer"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

raw = drop_empty_rows(rename_columns(pd.read_csv(TRAIN)))  # before cleaning
df = drop_empty_rows(clean(pd.read_csv(TRAIN)))            # after cleaning
