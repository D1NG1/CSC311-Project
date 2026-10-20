"""Price: formats, parsing results, skew, outliers, missing values."""
import numpy as np
import pandas as pd

from eda_common import df, raw

text = raw["price"].astype("string").str.strip()
plain = pd.to_numeric(text, errors="coerce").notna()
print(f"answers {text.notna().sum()} | plain numbers {plain.sum()} | other formats {(text.notna() & ~plain).sum()}")
print("examples:", text[text.notna() & ~plain].sample(8, random_state=0).tolist())

p = df["price"]
print(f"parsed as 0: {(p == 0).sum()} | missing {p.isna().sum()} ({p.isna().mean():.1%})")
print("unparseable:", text[text.notna() & p.isna()].str.slice(0, 40).tolist())
print(p.describe(percentiles=[0.5, 0.95, 0.99]).round(0))
print("skew raw", round(p.skew(), 1), "| skew log", round(np.log1p(p).skew(), 1))
print("median by class:", p.groupby(df["label"]).median().to_dict())
print("missing by class:", p.isna().groupby(df["label"]).mean().round(3).to_dict())

q1, q3 = p.quantile([0.25, 0.75])
print("IQR outliers (> {:g}): {}".format(q3 + 1.5 * (q3 - q1), (p > q3 + 1.5 * (q3 - q1)).sum()))
