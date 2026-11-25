"""
Tune L2-regularized Softmax Regression via 5-fold grouped CV.
Run from src/: uv run python -m training.train_softmax
"""
from sklearn.linear_model import LogisticRegression

from training import train_common

MODEL = LogisticRegression(solver="lbfgs", max_iter=2000)
PARAM_GRID = train_common.FEATURE_GRID | {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}


def main():
    """Tune Softmax Regression across all feature configs and L2 strengths (C = 1 / lambda)."""
    df = train_common.load_train_df()
    results = train_common.grid_search(df, MODEL, PARAM_GRID)

    print("\n=== Top 10 Softmax Regression Configs ===")
    train_common.print_top_results(results)

    print("\n=== Best Softmax Regression ===")
    train_common.print_top_results(results, top_k=1)

    csv_path = train_common.save_results("softmax_cv", results)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
