"""ease: distribution, missing values, outliers."""
import pandas as pd

from eda_common import df

s = df["ease"]
print(s.describe().round(2))
print("skew", round(s.skew(), 2), "| non-integer", s[s % 1 != 0].tolist())
print(s.value_counts().sort_index())
print(df.groupby("label")["ease"].agg(["mean", "median", "std"]).round(2))
print(pd.crosstab(df["label"], pd.cut(s, [-1, 3, 6, 10]), normalize="index").round(2))

q1, q3 = s.quantile([0.25, 0.75])
print("missing", s.isna().sum(), "| IQR outliers", ((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
