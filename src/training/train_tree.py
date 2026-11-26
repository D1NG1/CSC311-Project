"""
Tune a Decision Tree and a Random Forest via 5-fold grouped CV.
Run from src/: uv run python -m training.train_tree
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from training import train_common

# Trees split on thresholds, so monotone rescaling (rating scaling, log price) cannot change them.
TREE_FEATURE_GRID = train_common.FEATURE_GRID | {
    "features__rating": ["raw"],
    "features__log_price": [False],
}

TREE = DecisionTreeClassifier(random_state=train_common.SEED)
TREE_GRID = TREE_FEATURE_GRID | {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 5, 10, 15, None],
    "model__min_samples_leaf": [1, 2, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10, 20],
}

FOREST = RandomForestClassifier(random_state=train_common.SEED, n_jobs=-1)
FOREST_GRID = TREE_FEATURE_GRID | {
    "model__n_estimators": [50, 100, 300],
    "model__max_features": ["sqrt", "log2"],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_leaf": [1, 2, 5],
}


def tune(df, name, label, model, param_grid):
    """Grid-search one model, print the top configs and save results/{name}.csv."""
    results = train_common.grid_search(df, model, param_grid)

    print(f"\n=== Top 10 {label} Configs ===")
    train_common.print_top_results(results)

    print(f"\n=== Best {label} ===")
    train_common.print_top_results(results, top_k=1)

    csv_path = train_common.save_results(name, results)
    print(f"\nSaved results to {csv_path}")


def main():
    """Tune a single Decision Tree and a Random Forest on the same folds."""
    df = train_common.load_train_df()
    tune(df, "tree_cv", "Decision Tree", TREE, TREE_GRID)
    tune(df, "forest_cv", "Random Forest", FOREST, FOREST_GRID)


if __name__ == "__main__":
    main()
