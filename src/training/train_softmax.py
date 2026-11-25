"""Tune L2-regularized Softmax Regression via 5-fold grouped CV."""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training import train_common

MODEL = LogisticRegression(solver="lbfgs", max_iter=2000)
PARAM_GRID = train_common.FEATURE_GRID | {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}


def main():
    """Tune Softmax Regression across all feature configs and L2 strengths (C = 1 / lambda)."""
    df = train_common.load_train_df()
    results = train_common.grid_search(df, MODEL, PARAM_GRID)
    best = results.iloc[0]

    print("\n=== Top 10 Softmax Regression Configs ===")
    train_common.print_top_results(results)

    print(
        f"\n=== Best Softmax Regression ===\n"
        f"Features:     rating={best['rating']}, likert={best['likert']}, "
        f"log_price={best['log_price']}, use_strategy={best['use_strategy']}\n"
        f"Hyperparams:  C={best['C']:g} (lambda={1 / best['C']:g})\n"
        f"Val Accuracy: {best['val_acc']:.4f} +/- {best['val_acc_std']:.4f}\n"
        f"Val Macro-F1: {best['val_f1']:.4f}\n"
        f"Train Acc:    {best['train_acc']:.4f}"
    )

    csv_path = train_common.save_results("softmax_cv", results)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
