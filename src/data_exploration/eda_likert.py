"""Likert: distribution by class, correlation, missing values. Saves likert_by_class.pdf."""
import matplotlib.pyplot as plt
import pandas as pd

from eda_common import CLASSES, FIG_DIR, df, raw
from config import LIKERT_COLS, LIKERT_MAP

for col in LIKERT_COLS:
    print(col, "| missing", df[col].isna().sum(), "| unexpected", set(raw[col].dropna()) - set(LIKERT_MAP))
    print(pd.crosstab(df["label"], df[col], normalize="index").mul(100).round(0))
print(df[["ease", *LIKERT_COLS]].corr().round(2))

# Figure: % of each answer within each class, 100% stacked bars
levels = range(1, 6)
colours = ["#c9403f", "#ef9a99", "#d6d5d0", "#86b6ef", "#2a78d6"]  # disagree (red) -> agree (blue)
fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.7), sharey=True)
for ax, col in zip(axes, LIKERT_COLS):
    shares = pd.crosstab(df["label"], df[col], normalize="index").reindex(
        index=CLASSES, columns=levels, fill_value=0) * 100
    lefts = shares.cumsum(axis=1) - shares
    for level, colour in zip(levels, colours):
        ax.barh(CLASSES, shares[level], left=lefts[level], height=0.75, color=colour, edgecolor="white")
        for cls, w in shares[level].items():
            if w >= 12:  # only label segments wide enough for the text
                ax.text(lefts.at[cls, level] + w / 2, cls, f"{w:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if level in (1, 5) else "black")
    ax.set_title(col, loc="left")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
axes[0].invert_yaxis()  # Baseball on top
axes[1].set_xlabel("% of responses in each class", fontsize=8)
fig.legend(list(LIKERT_MAP), loc="upper center", ncol=5, frameon=False, fontsize=7.5,
           handlelength=1, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FIG_DIR / "likert_by_class.pdf", bbox_inches="tight")
