"""Training-size sweep for conformalized prediction intervals across Ridge, Tree, and Zoo CNN.

Uses a fixed game-level split (train_pool/calibration/test), nested train subsets in 1,000-play
steps, and conformal interval padding on top of model central intervals from predictive distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import tensorflow as tf

from run_zoo import get_conv_net


MIN_IDX_Y = 71
MAX_IDX_Y = 150
NUM_CLASSES = MAX_IDX_Y - MIN_IDX_Y + 1
ALL_CLASSES = np.arange(NUM_CLASSES)


@dataclass
class SplitData:
    train_pool_idx: np.ndarray
    calibration_idx: np.ndarray
    test_idx: np.ndarray


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(processed_dir: Path, raw_train_csv: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    train_x = np.load(processed_dir / "train_x.npy")
    train_y = pd.read_pickle(processed_dir / "train_y.pkl")
    raw = pd.read_csv(raw_train_csv, usecols=["GameId", "PlayId"])
    raw["PlayId"] = raw["PlayId"].astype(str)
    gcount = raw.groupby("PlayId")["GameId"].nunique()
    bad = gcount[gcount > 1]
    if len(bad) > 0:
        raise ValueError(
            "PlayId maps to multiple GameIds. Rebuild processed data using (GameId, PlayId) keys."
        )
    raw = raw.drop_duplicates("PlayId")

    play_map = train_y[["PlayId"]].copy()
    play_map["PlayId"] = play_map["PlayId"].astype(str)
    play_map["BasePlayId"] = play_map["PlayId"].str.replace("_aug", "", regex=False)
    play_map = play_map.merge(raw.rename(columns={"PlayId": "BasePlayId"}), on="BasePlayId", how="left")
    if play_map["GameId"].isna().any():
        raise ValueError("Missing GameId mapping for some plays. Ensure raw train.csv matches processed data.")
    return train_x, train_y, play_map


def build_game_split(play_map: pd.DataFrame, seed: int, train_frac: float = 0.6, cal_frac: float = 0.2) -> SplitData:
    rng = np.random.default_rng(seed)
    games = np.array(sorted(play_map["GameId"].unique()))
    rng.shuffle(games)

    n_games = len(games)
    n_train = int(n_games * train_frac)
    n_cal = int(n_games * cal_frac)

    train_games = set(games[:n_train])
    cal_games = set(games[n_train : n_train + n_cal])
    test_games = set(games[n_train + n_cal :])

    g = play_map["GameId"].to_numpy()
    return SplitData(
        train_pool_idx=np.where(np.isin(g, list(train_games)))[0],
        calibration_idx=np.where(np.isin(g, list(cal_games)))[0],
        test_idx=np.where(np.isin(g, list(test_games)))[0],
    )


def make_tabular_features(train_x: np.ndarray) -> np.ndarray:
    flat = train_x.reshape(train_x.shape[0], -1, train_x.shape[-1])
    mins = flat.min(axis=1)
    means = flat.mean(axis=1)
    maxs = flat.max(axis=1)
    stds = flat.std(axis=1)
    return np.concatenate([mins, means, maxs, stds], axis=1)


def one_hot_labels(yard_idx_clipped: np.ndarray) -> np.ndarray:
    y = yard_idx_clipped - MIN_IDX_Y
    out = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def central_interval_from_proba(proba: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(proba, axis=1)
    lower = np.argmax(cdf >= (alpha / 2.0), axis=1)
    upper = np.argmax(cdf >= (1.0 - alpha / 2.0), axis=1)
    return lower.astype(int), upper.astype(int)


def conformal_padding(cal_y: np.ndarray, cal_l: np.ndarray, cal_u: np.ndarray, alpha: float) -> int:
    scores = np.maximum.reduce([cal_l - cal_y, cal_y - cal_u, np.zeros_like(cal_y)])
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q = int(np.partition(scores, k - 1)[k - 1])
    return q


def evaluate_with_conformal(
    cal_proba: np.ndarray,
    test_proba: np.ndarray,
    cal_y: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    cal_l, cal_u = central_interval_from_proba(cal_proba, alpha)
    test_l, test_u = central_interval_from_proba(test_proba, alpha)

    q = conformal_padding(cal_y, cal_l, cal_u, alpha)

    lo = np.maximum(0, test_l - q)
    hi = np.minimum(NUM_CLASSES - 1, test_u + q)
    width = hi - lo
    covered = (test_y >= lo) & (test_y <= hi)

    return {
        "mean_width": float(np.mean(width)),
        "coverage": float(np.mean(covered)),
        "q_padding": float(q),
    }


def _proba_to_num_classes(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    """Expand predict_proba to (n, NUM_CLASSES); proba column j is class model_classes[j]."""
    out = np.zeros((proba.shape[0], NUM_CLASSES), dtype=np.float64)
    for j, c in enumerate(model_classes):
        if 0 <= c < NUM_CLASSES:
            out[:, c] = proba[:, j]
    return out


def fit_ridge_fixed_classes(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    seed: int,
    batch_size: int = 64,
    n_epochs: int = 50,
) -> SGDClassifier:
    """Fit SGDClassifier over fixed class set so proba always has NUM_CLASSES columns."""
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        random_state=seed,
        max_iter=1,
        warm_start=False,
    )
    n = len(x)
    rng = np.random.default_rng(seed)
    first_batch = True
    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) == 0:
                continue
            X_batch = x[idx]
            y_batch = y[idx]
            if first_batch:
                clf.partial_fit(X_batch, y_batch, classes=ALL_CLASSES)
                first_batch = False
            else:
                clf.partial_fit(X_batch, y_batch)
    return clf


def tune_ridge(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    n = min(5000, len(x))
    x_dev = x[:n]
    y_dev = y[:n]
    # SGDClassifier uses alpha (reg strength); map from C (inverse reg): alpha = 1/C
    alphas = [1.0 / c for c in (0.1, 0.3, 1.0, 3.0)]
    best_alpha = alphas[0]
    best_nll = float("inf")
    for alpha in alphas:
        model = fit_ridge_fixed_classes(x_dev, y_dev, alpha=alpha, seed=seed)
        proba = _proba_to_num_classes(model.predict_proba(x_dev), model.classes_)
        nll = log_loss(y_dev, proba, labels=ALL_CLASSES)
        if nll < best_nll:
            best_nll, best_alpha = nll, alpha
    return best_alpha


def fit_tree_fixed_classes(
    x: np.ndarray, y: np.ndarray, params: dict, seed: int
) -> lgb.LGBMClassifier:
    """Single LightGBM multiclass fit with num_class=NUM_CLASSES so proba always has 80 columns."""
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASSES,
        n_estimators=200,
        random_state=seed,
        verbose=-1,
        **params,
    )
    model.fit(x, y)
    return model


def tune_tree(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    n = min(5000, len(x))
    x_dev = x[:n]
    y_dev = y[:n]
    grid = [
        {"learning_rate": 0.05, "max_depth": 6},
        {"learning_rate": 0.05, "max_depth": 10},
        {"learning_rate": 0.1, "max_depth": 6},
    ]
    best = grid[0]
    best_nll = float("inf")
    for params in grid:
        model = fit_tree_fixed_classes(x_dev, y_dev, params, seed=seed)
        proba = _proba_to_num_classes(model.predict_proba(x_dev), model.classes_)
        nll = log_loss(y_dev, proba, labels=ALL_CLASSES)
        if nll < best_nll:
            best_nll, best = nll, params
    return best


def train_cnn(train_x: np.ndarray, y_train: np.ndarray, seed: int, epochs: int = 8) -> tf.keras.Model:
    tf.random.set_seed(seed)
    model = get_conv_net(NUM_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    model.fit(train_x, one_hot_labels(y_train + MIN_IDX_Y), epochs=epochs, batch_size=64, verbose=0)
    return model


def run_sweep(
    alpha: float,
    seed: int,
    out_dir: Path,
    processed_dir: Path,
    raw_csv: Path,
    toy: bool = False,
) -> None:
    set_determinism(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, play_map = load_data(processed_dir, raw_csv)
    # Drop _aug plays so cal/test and play counts are on original plays only
    is_aug = play_map["PlayId"].str.endswith("_aug").to_numpy()
    keep = ~is_aug
    train_x = train_x[keep]
    train_y = train_y.iloc[keep].reset_index(drop=True)
    play_map = play_map.iloc[keep].reset_index(drop=True)

    tab_x = make_tabular_features(train_x)
    y_idx = train_y["YardIndexClipped"].to_numpy(dtype=int)
    y_cls = y_idx - MIN_IDX_Y

    split = build_game_split(play_map, seed=seed)
    train_pool = split.train_pool_idx
    cal_idx = split.calibration_idx
    test_idx = split.test_idx

    rng = np.random.default_rng(seed)
    perm_train_pool = rng.permutation(train_pool)

    ridge_alpha = tune_ridge(tab_x[perm_train_pool], y_cls[perm_train_pool], seed)
    tree_params = tune_tree(tab_x[perm_train_pool], y_cls[perm_train_pool], seed)

    n_max = (len(train_pool) // 1000) * 1000
    sizes = list(range(1000, n_max + 1, 1000))
    if toy:
        sizes = sizes[:2]
        cnn_epochs = 2
    else:
        cnn_epochs = 8
    rows: list[dict[str, float | int | str]] = []

    for n_train in sizes:
        subset = perm_train_pool[:n_train]

        ridge = fit_ridge_fixed_classes(
            tab_x[subset], y_cls[subset], alpha=ridge_alpha, seed=seed
        )
        ridge_cal = _proba_to_num_classes(ridge.predict_proba(tab_x[cal_idx]), ridge.classes_)
        ridge_test = _proba_to_num_classes(ridge.predict_proba(tab_x[test_idx]), ridge.classes_)
        ridge_metrics = evaluate_with_conformal(ridge_cal, ridge_test, y_cls[cal_idx], y_cls[test_idx], alpha)
        ridge_metrics["nll"] = float(log_loss(y_cls[test_idx], ridge_test, labels=ALL_CLASSES))

        tree = fit_tree_fixed_classes(tab_x[subset], y_cls[subset], tree_params, seed=seed)
        tree_cal = _proba_to_num_classes(tree.predict_proba(tab_x[cal_idx]), tree.classes_)
        tree_test = _proba_to_num_classes(tree.predict_proba(tab_x[test_idx]), tree.classes_)
        tree_metrics = evaluate_with_conformal(tree_cal, tree_test, y_cls[cal_idx], y_cls[test_idx], alpha)
        tree_metrics["nll"] = float(log_loss(y_cls[test_idx], tree_test, labels=ALL_CLASSES))

        cnn_model = train_cnn(train_x[subset], y_cls[subset], seed, epochs=cnn_epochs)
        cnn_cal = cnn_model.predict(train_x[cal_idx], verbose=0)
        cnn_test = cnn_model.predict(train_x[test_idx], verbose=0)
        cnn_metrics = evaluate_with_conformal(cnn_cal, cnn_test, y_cls[cal_idx], y_cls[test_idx], alpha)
        cnn_metrics["nll"] = float(log_loss(y_cls[test_idx], cnn_test, labels=ALL_CLASSES))

        for model_name, metrics in [
            ("ridge_multinomial", ridge_metrics),
            ("tree_multiclass_hgbt", tree_metrics),
            ("zoo_cnn_distributional", cnn_metrics),
        ]:
            rows.append(
                {
                    "model": model_name,
                    "n_train": n_train,
                    "mean_width": metrics["mean_width"],
                    "coverage": metrics["coverage"],
                    "nll": metrics["nll"],
                    "q_padding": metrics["q_padding"],
                    "seed": seed,
                    "alpha": alpha,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        print(f"Done n_train={n_train}")

    results = pd.DataFrame(rows)
    results_path = out_dir / "training_size_sweep_results.csv"
    results.to_csv(results_path, index=False)

    plot_path = out_dir / "training_size_sweep.png"
    make_plot(results, alpha=alpha, output_path=plot_path)

    readme_path = out_dir / "experiment_README.md"
    write_experiment_readme(readme_path, alpha=alpha, seed=seed, ridge_alpha=ridge_alpha, tree_params=tree_params)

    config_path = out_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "alpha": alpha,
                "label_support": [MIN_IDX_Y, MAX_IDX_Y],
                "split": "gameId: train_pool=60%, calibration=20%, test=20%",
                "nested_subsets": "first n plays from one seeded permutation",
            },
            indent=2,
        )
    )
    print(f"Saved results to {results_path} and figure to {plot_path}")


def make_plot(results: pd.DataFrame, alpha: float, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for model_name, group in results.groupby("model"):
        g = group.sort_values("n_train")
        axes[0].plot(g["n_train"], g["mean_width"], marker="o", label=model_name)
        axes[1].plot(g["n_train"], g["coverage"], marker="o", label=model_name)

    axes[0].set_ylabel("Mean conformal interval width")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Training plays")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].axhline(1 - alpha, color="black", linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_experiment_readme(
    path: Path, alpha: float, seed: int, ridge_alpha: float, tree_params: dict[str, float]
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Training-size conformal sweep",
                "",
                "- Split scheme: by `GameId` with fixed partitions train_pool=60%, calibration=20%, test=20%.",
                f"- Miscoverage level: alpha={alpha} (target coverage={1-alpha:.2f}).",
                f"- Label support: YardIndexClipped in [{MIN_IDX_Y}, {MAX_IDX_Y}] ({NUM_CLASSES} classes).",
                "- Conformal method: interval-from-distribution central interval + calibration padding q via finite-sample conformal quantile.",
                f"- Seed: {seed}.",
                f"- Frozen hyperparameters: Ridge (SGD) alpha={ridge_alpha}; Tree (LightGBM multiclass, num_class={NUM_CLASSES}) params={tree_params}; CNN from `run_zoo.get_conv_net` with fixed optimizer/training schedule.",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training-size conformal stability sweep.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--toy", action="store_true", help="Quick run: 2 training sizes, 2 CNN epochs. Use to verify pipeline and outputs.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/training_size_sweep"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        alpha=args.alpha,
        seed=args.seed,
        out_dir=args.out_dir,
        processed_dir=args.processed_dir,
        raw_csv=args.raw_train_csv,
        toy=args.toy,
    )
