"""Multi-select: % of each class choosing each option, unknown options. Saves multi_by_class.pdf."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from eda_common import CLASSES, FIG_DIR, df, raw
from config import MULTI_COLS, MULTI_OPTIONS
from preprocess import parse_multi_col

# Shorter option names for the figure
SHORT = {
    "Indoor facility": "Indoor",
    "Outdoor public space": "Outdoor public",
    "School or university facility": "School",
    "Community recreation facility": "Community",
    "Private sports facility": "Private",
    "At home or nearby": "At home",
    "Hip-hop or rap": "Rap",
    "Electronic or dance": "Electronic",
    "Folk or traditional": "Traditional",
    "Classical or jazz": "Jazz",
    "No particular type": "No particular",
}


def shares(col):
    """% of each class choosing each option (rows: classes)."""
    s = parse_multi_col(df[col], col).groupby(df["label"]).mean().reindex(CLASSES) * 100
    s.columns = MULTI_OPTIONS[col]
    return s


for col in MULTI_COLS:
    unknown = set(raw[col].dropna().str.split(",").explode()) - set(MULTI_OPTIONS[col])
    print(col, "| missing", df[col].isna().sum(), "| unknown options", unknown)
    t = shares(col).T
    t["spread"] = t.max(axis=1) - t.min(axis=1)  # small spread = not informative
    print(t.round(0).sort_values("spread", ascending=False))

# Figure: heatmap, one panel per question
cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#86b6ef", "#2a78d6", "#104281"])
fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.9),
                         gridspec_kw={"width_ratios": [len(MULTI_OPTIONS[c]) for c in MULTI_COLS]})
for ax, col in zip(axes, MULTI_COLS):
    s = shares(col)
    n_rows, n_cols = s.shape
    ax.pcolormesh(np.arange(n_cols + 1) - 0.5, np.arange(n_rows + 1) - 0.5, s.values,
                  cmap=cmap, vmin=0, vmax=100, edgecolors="white", linewidth=1.5)
    for (i, j), v in np.ndenumerate(s.values):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v >= 55 else "black")
    ax.set_xticks(range(n_cols), [SHORT.get(o, o) for o in s.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows), CLASSES if col == MULTI_COLS[0] else [])
    ax.set_ylim(n_rows - 0.5, -0.5)  # Baseball on top
    ax.set_title(col, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG_DIR / "multi_by_class.pdf", bbox_inches="tight")
