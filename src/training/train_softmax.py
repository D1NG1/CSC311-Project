"""Train and tune L2-regularized Softmax Regression via 5-fold grouped CV."""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training import train_common

LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def make_softmax(lam):
    """Return Softmax Regression where C = 1 / lam sets L2 penalty strength."""
    return LogisticRegression(C=1.0 / lam, solver="lbfgs", max_iter=2000)


def main():
    """Tune L2-regularized Softmax Regression across feature and lambda grids."""
    df = train_common.load_train_df()
    folds = train_common.get_folds(df)
    feature_configs = train_common.feature_grid()
    param_grid = [{"lam": lam} for lam in LAMBDAS]

    rows = train_common.evaluate_grid(df, folds, feature_configs, param_grid, make_softmax)
    best = max(rows, key=train_common.score_key)

    print("\n=== Top 10 Softmax Regression Configs ===")
    train_common.print_top_results(rows, top_k=10)

    features_str = ", ".join(f"{k}={best[k]}" for k in feature_configs[0])
    print(
        f"\n=== Best Softmax Regression ===\n"
        f"Features:     {features_str}\n"
        f"Hyperparams:  lam={best['lam']:g} (C={1.0 / best['lam']:g})\n"
        f"Val Accuracy: {best['val_acc']:.4f} +/- {best['val_acc_std']:.4f}\n"
        f"Val Macro-F1: {best['val_f1']:.4f}\n"
        f"Train Acc:    {best['train_acc']:.4f}"
    )

    csv_path = train_common.save_results("softmax_cv", rows)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
