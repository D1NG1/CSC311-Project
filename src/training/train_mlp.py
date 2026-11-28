"""
Tune a one-hidden-layer MLP via 5-fold grouped CV.
Run from src/: uv run python -m training.train_mlp
"""
from sklearn.neural_network import MLPClassifier

from training import train_common

# Feature settings fixed to the best softmax config: the full FEATURE_GRID times the MLP grid
# would be ~14k configs. The MLP sees the same features as softmax, so the comparison stays fair
MLP_FEATURE_GRID = {
    "features__likert": ["onehot"],
    "features__rating": ["divide"],
    "features__log_price": [False],
    "features__use_strategy": [True],
}

MODEL = MLPClassifier(solver="adam", activation="relu", random_state=train_common.SEED)
PARAM_GRID = MLP_FEATURE_GRID | {
    "model__hidden_layer_sizes": [(16,), (32,), (64,), (128,)], #single layer of hidden units
    "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],  #L2 penalty
    "model__learning_rate_init": [1e-4, 1e-3, 1e-2],
    "model__batch_size": [32, 64, 128],
    "model__max_iter": [50, 100, 200, 400],  #max epochs for adam
}


def main():
    """Tune the MLP across hidden layer size, alpha / L2, learning rate, batch size and epochs."""
    df = train_common.load_train_df()
    results = train_common.grid_search(df, MODEL, PARAM_GRID)

    print("\n=== Top 10 MLP Configs ===")
    train_common.print_top_results(results)

    print("\n=== Best MLP ===")
    train_common.print_top_results(results, top_k=1)

    csv_path = train_common.save_results("mlp_cv", results)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()








