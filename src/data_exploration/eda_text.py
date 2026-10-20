"""Free text: missing values by class, distinctive words per class."""
import pandas as pd

from eda_common import CLASSES, df, raw
from config import TEXT_COLS
from text import tokenize

STOPWORDS = set("a an and the of to for in on with by i it is me my you but or this that be at so very s t we when".split())

for col in TEXT_COLS:
    print(f"{col} | missing {df[col].isna().mean():.1%} (empty {raw[col].isna().sum()}) | by class",
          df[col].isna().groupby(df["label"]).mean().round(3).to_dict())
    print(f"  distinct answers: raw {raw[col].nunique()}, cleaned {df[col].nunique()}")

    # % of each class's answers containing each word; distinctive = this class minus the others' mean
    words = df[col].map(lambda t: set(tokenize(t)) - STOPWORDS)
    share = pd.DataFrame({c: words[df["label"] == c].explode().value_counts() / (df["label"] == c).sum() * 100
                          for c in CLASSES}).fillna(0)
    share = share[share.sum(axis=1) >= 5]  # drop rare words
    for c in CLASSES:
        score = share[c] - share.drop(columns=c).mean(axis=1)
        print(f"  {c:10s}", ", ".join(f"{w} ({share.at[w, c]:.0f}%)" for w in score.nlargest(3).index))
