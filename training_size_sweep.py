"""Training-size sweep for conformalized prediction intervals across Ridge, Tree, and Zoo CNN.

Uses a fixed game-level split (train_pool/calibration/test), nested train subsets in 20-game
chunks, and conformal interval padding on top of model central intervals from predictive distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import tensorflow as tf

from run_zoo import get_conv_net, crps, Metric
from tensorflow.keras.callbacks import EarlyStopping


MIN_IDX_Y = 71
MAX_IDX_Y = 150
NUM_CLASSES = MAX_IDX_Y - MIN_IDX_Y + 1
ALL_CLASSES = np.arange(NUM_CLASSES)
TUNE_VAL_GAMES = 40  # Games held out from train pool for hyperparameter tuning


@dataclass
class SplitData:
    train_pool_idx: np.ndarray
    calibration_idx: np.ndarray
    test_idx: np.ndarray
    train_games_ordered: np.ndarray  # Shuffled train-pool game IDs for nested subsets


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(processed_dir: Path, raw_train_csv: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load processed features/labels and attach GameId via raw CSV; assert PlayId unique per game."""
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
    """Split by game (60% train pool, 20% cal, 20% test) so cal/test are fixed across n_train."""
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
    train_games_ordered = games[:n_train]  # First n_train games after shuffle
    return SplitData(
        train_pool_idx=np.where(np.isin(g, list(train_games)))[0],
        calibration_idx=np.where(np.isin(g, list(cal_games)))[0],
        test_idx=np.where(np.isin(g, list(test_games)))[0],
        train_games_ordered=train_games_ordered,
    )


def _tabular_df(x: np.ndarray) -> pd.DataFrame:
    """Wrap tabular array as DataFrame with feature names so LGBM doesn't warn."""
    n_feat = x.shape[1] if x.ndim > 1 else x.size
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(n_feat)])


def make_tabular_features(train_x: np.ndarray) -> np.ndarray:
    """Flatten spatial dims and aggregate min/mean/max/std for ridge and tree."""
    flat = train_x.reshape(train_x.shape[0], -1, train_x.shape[-1])
    mins = flat.min(axis=1)
    means = flat.mean(axis=1)
    maxs = flat.max(axis=1)
    stds = flat.std(axis=1)
    return np.concatenate([mins, means, maxs, stds], axis=1)


def crps_numpy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """CRPS for ordinal outcome: L2 on cumulative distributions (scaled by 199, match run_zoo)."""
    proba = normalize_proba_rows(proba)
    n = len(y_true)
    y_oh = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    y_oh[np.arange(n), y_true.astype(int)] = 1.0
    cdf_pred = np.cumsum(proba, axis=1)
    cdf_true = np.cumsum(y_oh, axis=1)
    return float(np.mean(np.sum((cdf_pred - cdf_true) ** 2, axis=1)) / 199)


def one_hot_labels(yard_idx_clipped: np.ndarray) -> np.ndarray:
    y = yard_idx_clipped - MIN_IDX_Y
    out = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def central_interval_from_proba(proba: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """(alpha/2, 1-alpha/2) quantiles of predictive CDF per sample → lower/upper bin indices."""
    proba = normalize_proba_rows(proba)
    cdf = np.cumsum(proba, axis=1)
    lower = np.argmax(cdf >= (alpha / 2.0), axis=1)
    upper = np.argmax(cdf >= (1.0 - alpha / 2.0), axis=1)
    return lower.astype(int), upper.astype(int)


def conformal_padding(cal_y: np.ndarray, cal_l: np.ndarray, cal_u: np.ndarray, alpha: float) -> int:
    """Nonconformity scores on cal set; return empirical (1-alpha) quantile as padding q."""
    scores = np.maximum.reduce([cal_l - cal_y, cal_y - cal_u, np.zeros_like(cal_y)])
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q = int(np.partition(scores, k - 1)[k - 1])
    return q


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile with nonnegative weights and q in [0, 1]."""
    if len(values) == 0:
        return 0.0
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if np.sum(w) <= 0:
        w = np.ones_like(w)

    order = np.argsort(values)
    v = np.asarray(values, dtype=np.float64)[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cdf, np.clip(q, 0.0, 1.0), side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _locality_features_from_proba(proba: np.ndarray) -> np.ndarray:
    """Construct low-dim locality features from predictive distributions."""
    p = normalize_proba_rows(proba)
    cls = np.arange(NUM_CLASSES, dtype=np.float64)
    mean = p @ cls
    second = p @ (cls**2)
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    return np.stack([mean, std, entropy], axis=1)


def local_conformal_padding(
    cal_y: np.ndarray,
    cal_l: np.ndarray,
    cal_u: np.ndarray,
    cal_feat: np.ndarray,
    test_feat: np.ndarray,
    alpha: float,
    local_k: int,
) -> np.ndarray:
    """Tibshirani-style locally weighted conformal padding q(x) for each test point."""
    scores = np.maximum.reduce([cal_l - cal_y, cal_y - cal_u, np.zeros_like(cal_y)]).astype(np.float64)

    # Standardize locality features so one coordinate cannot dominate distances.
    feat_scale = np.std(cal_feat, axis=0, ddof=1)
    feat_scale = np.where(feat_scale > 1e-8, feat_scale, 1.0)
    cal_scaled = cal_feat / feat_scale
    test_scaled = test_feat / feat_scale

    n_cal = len(cal_scaled)
    k = min(max(int(local_k), 5), n_cal)
    qhat = np.zeros(len(test_scaled), dtype=np.float64)
    target_q = min(1.0, (1.0 - alpha) * (n_cal + 1) / n_cal)

    for j in range(len(test_scaled)):
        d2 = np.sum((cal_scaled - test_scaled[j]) ** 2, axis=1)
        # Adaptive bandwidth from kth-nearest calibration point.
        kth = np.partition(d2, k - 1)[k - 1]
        bw = float(np.sqrt(max(kth, 1e-12)))
        w = np.exp(-0.5 * d2 / (bw**2))
        qhat[j] = _weighted_quantile(scores, w, target_q)

    return np.ceil(qhat).astype(int)


def evaluate_with_conformal(
    cal_proba: np.ndarray,
    test_proba: np.ndarray,
    cal_y: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
    local_k: int,
) -> dict[str, float]:
    """Central interval from model + locally weighted conformal q(x) on cal."""
    cal_l, cal_u = central_interval_from_proba(cal_proba, alpha)
    test_l, test_u = central_interval_from_proba(test_proba, alpha)
    cal_feat = _locality_features_from_proba(cal_proba)
    test_feat = _locality_features_from_proba(test_proba)

    q = local_conformal_padding(cal_y, cal_l, cal_u, cal_feat, test_feat, alpha, local_k=local_k)

    lo = np.maximum(0, test_l - q)
    hi = np.minimum(NUM_CLASSES - 1, test_u + q)
    width = hi - lo
    covered = (test_y >= lo) & (test_y <= hi)

    n_test = len(width)
    mean_width = float(np.mean(width))
    std_width = float(np.std(width, ddof=1)) if n_test > 1 else 0.0
    se_width = std_width / (n_test**0.5)
    coverage = float(np.mean(covered))

    return {
        "mean_width": mean_width,
        "std_width": std_width,
        "se_width": se_width,
        "n_test": float(n_test),
        "coverage": coverage,
        "mean_q": float(np.mean(q)),
        "std_q": float(np.std(q, ddof=1)) if len(q) > 1 else 0.0,
    }


def normalize_proba_rows(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-normalized probabilities for stable metrics and interval construction."""
    p = np.asarray(proba, dtype=np.float64)
    p = np.clip(p, eps, None)
    row_sums = p.sum(axis=1, keepdims=True)
    # Guard against pathological all-zero/NaN rows from upstream models.
    bad = ~np.isfinite(row_sums) | (row_sums <= 0)
    if np.any(bad):
        p[bad[:, 0]] = 1.0 / p.shape[1]
        row_sums = p.sum(axis=1, keepdims=True)
    return p / row_sums


def _proba_to_num_classes(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    """Expand predict_proba to (n, NUM_CLASSES); proba column j is class model_classes[j]."""
    out = np.zeros((proba.shape[0], NUM_CLASSES), dtype=np.float64)
    for j, c in enumerate(model_classes):
        if 0 <= c < NUM_CLASSES:
            out[:, c] = proba[:, j]
    return normalize_proba_rows(out)


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


def tune_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> float:
    """Pick ridge alpha (1/C) by best CRPS on tune_val (40 games held out from train pool)."""
    alphas = [1.0 / c for c in (0.1, 0.3, 1.0, 3.0)]
    best_alpha = alphas[0]
    best_crps = float("inf")
    for alpha in alphas:
        model = fit_ridge_fixed_classes(x_train, y_train, alpha=alpha, seed=seed)
        proba = _proba_to_num_classes(model.predict_proba(x_val), model.classes_)
        crps = crps_numpy(y_val, proba)
        if crps < best_crps:
            best_crps, best_alpha = crps, alpha
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
    model.fit(_tabular_df(x), y)
    return model


def tune_tree(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> dict[str, float]:
    """Pick hyperparameters by best CRPS on tune_val (40 games held out from train pool)."""
    grid = [
        {"learning_rate": 0.05, "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
        {"learning_rate": 0.05, "max_depth": 7, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
        {"learning_rate": 0.1, "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
        {"learning_rate": 0.05, "max_depth": 5, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 0.5},
    ]
    best = grid[0]
    best_crps = float("inf")
    for params in grid:
        model = fit_tree_fixed_classes(x_train, y_train, params, seed=seed)
        proba = _proba_to_num_classes(model.predict_proba(_tabular_df(x_val)), model.classes_)
        crps = crps_numpy(y_val, proba)
        if crps < best_crps:
            best_crps, best = crps, params
    return best


def train_cnn(train_x: np.ndarray, y_train: np.ndarray, seed: int, epochs: int = 50) -> tf.keras.Model:
    """Full Zoo: CRPS loss + early stopping on 20% val, matching run_zoo."""
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y_train)
    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_train = train_x[train_idx]
    X_val = train_x[val_idx]
    y_train_oh = one_hot_labels(y_train[train_idx] + MIN_IDX_Y).astype(np.float32)
    y_val_oh = one_hot_labels(y_train[val_idx] + MIN_IDX_Y).astype(np.float32)

    model = get_conv_net(NUM_CLASSES)
    es = EarlyStopping(
        monitor="val_CRPS", mode="min", restore_best_weights=True, verbose=0, patience=10
    )
    es.set_model(model)
    metric = Metric(model, [es], [X_val, y_val_oh])
    model.compile(loss=crps, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    model.fit(
        X_train,
        y_train_oh,
        epochs=epochs,
        batch_size=64,
        verbose=0,
        callbacks=[metric],
        validation_data=(X_val, y_val_oh),
    )
    return model


def run_sweep(
    alpha: float,
    seed: int,
    out_dir: Path,
    processed_dir: Path,
    raw_csv: Path,
    max_k: int = 50,
    local_k: int = 200,
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
    train_games_ordered = split.train_games_ordered
    cal_idx = split.calibration_idx
    test_idx = split.test_idx
    g = play_map["GameId"].to_numpy()

    # Hold out last TUNE_VAL_GAMES from train pool for hyperparameter tuning
    n_tune_val = min(TUNE_VAL_GAMES, len(train_games_ordered) - 20)  # Keep at least 20 for sweep
    sweep_train_games = train_games_ordered[: -n_tune_val]
    tune_val_games = train_games_ordered[-n_tune_val :]
    sweep_train_pool = train_pool[np.isin(g[train_pool], list(sweep_train_games))]
    tune_val_idx_arr = np.where(np.isin(g, list(tune_val_games)))[0]

    ridge_alpha = tune_ridge(
        tab_x[sweep_train_pool], y_cls[sweep_train_pool],
        tab_x[tune_val_idx_arr], y_cls[tune_val_idx_arr], seed
    )
    tree_params = tune_tree(
        tab_x[sweep_train_pool], y_cls[sweep_train_pool],
        tab_x[tune_val_idx_arr], y_cls[tune_val_idx_arr], seed
    )

    # Sizes 20, 40, 60, ... games (20-game chunks), up to max_k * 20 capped by sweep-train games
    max_games = min(max_k * 20, len(sweep_train_games))
    sizes = list(range(20, max_games + 1, 20))
    cnn_epochs = 50  # full Zoo (match run_zoo)
    output_stem = f"sweep_{max_k}"
    rows: list[dict[str, float | int | str]] = []

    for n_games in sizes:
        first_n_games = set(sweep_train_games[:n_games])
        subset = sweep_train_pool[np.isin(g[sweep_train_pool], list(first_n_games))]

        ridge = fit_ridge_fixed_classes(
            tab_x[subset], y_cls[subset], alpha=ridge_alpha, seed=seed
        )
        ridge_cal = _proba_to_num_classes(ridge.predict_proba(tab_x[cal_idx]), ridge.classes_)
        ridge_test = _proba_to_num_classes(ridge.predict_proba(tab_x[test_idx]), ridge.classes_)
        ridge_metrics = evaluate_with_conformal(
            ridge_cal, ridge_test, y_cls[cal_idx], y_cls[test_idx], alpha, local_k=local_k
        )
        ridge_metrics["crps"] = crps_numpy(y_cls[test_idx], ridge_test)

        tree = fit_tree_fixed_classes(tab_x[subset], y_cls[subset], tree_params, seed=seed)
        tree_cal = _proba_to_num_classes(
            tree.predict_proba(_tabular_df(tab_x[cal_idx])), tree.classes_
        )
        tree_test = _proba_to_num_classes(
            tree.predict_proba(_tabular_df(tab_x[test_idx])), tree.classes_
        )
        tree_metrics = evaluate_with_conformal(
            tree_cal, tree_test, y_cls[cal_idx], y_cls[test_idx], alpha, local_k=local_k
        )
        tree_metrics["crps"] = crps_numpy(y_cls[test_idx], tree_test)

        cnn_model = train_cnn(train_x[subset], y_cls[subset], seed, epochs=cnn_epochs)
        cnn_cal = cnn_model.predict(train_x[cal_idx], verbose=0)
        cnn_test = cnn_model.predict(train_x[test_idx], verbose=0)
        cnn_metrics = evaluate_with_conformal(
            cnn_cal, cnn_test, y_cls[cal_idx], y_cls[test_idx], alpha, local_k=local_k
        )
        cnn_metrics["crps"] = crps_numpy(y_cls[test_idx], cnn_test)

        for model_name, metrics in [
            ("ridge_multinomial", ridge_metrics),
            ("tree_multiclass_hgbt", tree_metrics),
            ("zoo_cnn_distributional", cnn_metrics),
        ]:
            rows.append(
                {
                    "model": model_name,
                    "n_train": n_games,
                    "n_plays": len(subset),
                    "mean_width": metrics["mean_width"],
                    "std_width": metrics["std_width"],
                    "se_width": metrics["se_width"],
                    "n_test": metrics["n_test"],
                    "coverage": metrics["coverage"],
                    "crps": metrics["crps"],
                    "mean_q": metrics["mean_q"],
                    "std_q": metrics["std_q"],
                    "seed": seed,
                    "alpha": alpha,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        print(f"Done n_train={n_games} games ({len(subset)} plays)")

    results = pd.DataFrame(rows)
    results_path = out_dir / f"{output_stem}.csv"
    results.to_csv(results_path, index=False)

    plot_path = out_dir / f"{output_stem}.png"
    make_plot(results, alpha=alpha, output_path=plot_path)

    readme_path = out_dir / "sweep_README.md"
    write_experiment_readme(readme_path, alpha=alpha, seed=seed, ridge_alpha=ridge_alpha, tree_params=tree_params, max_k=max_k, local_k=local_k)

    config_path = out_dir / f"{output_stem}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "max_k": max_k,
                "seed": seed,
                "alpha": alpha,
                "label_support": [MIN_IDX_Y, MAX_IDX_Y],
                "split": "gameId: train_pool=60%, calibration=20%, test=20%; 40 games held out from train for tune-val",
                "nested_subsets": "first N games (20-game chunks) from shuffled train pool",
                "conformal": "locally weighted conformal intervals (Tibshirani-style) over predictive-distribution locality features",
                "local_k": local_k,
            },
            indent=2,
        )
    )
    print(f"Saved results to {results_path} and figure to {plot_path}")


def make_plot(results: pd.DataFrame, alpha: float, output_path: Path) -> None:
    """Three panels: mean conformal width, empirical coverage, and CRPS vs n_train, with SE error bars."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for model_name, group in results.groupby("model"):
        g = group.sort_values("n_train")
        axes[0].errorbar(
            g["n_train"], g["mean_width"], yerr=g["se_width"], marker="o", label=model_name
        )
        # SE for proportion: sqrt(p*(1-p)/n)
        p = g["coverage"].to_numpy()
        n = g["n_test"].to_numpy()
        se_cov = np.sqrt(np.maximum(0, p * (1 - p) / n))
        axes[1].errorbar(g["n_train"], g["coverage"], yerr=se_cov, marker="o", label=model_name)
        axes[2].plot(g["n_train"], g["crps"], marker="o", label=model_name)

    axes[0].set_ylabel("Mean conformal interval width")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("Empirical coverage")
    axes[1].axhline(1 - alpha, color="black", linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("Training games")
    axes[2].set_ylabel("CRPS (test)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_experiment_readme(
    path: Path,
    alpha: float,
    seed: int,
    ridge_alpha: float,
    tree_params: dict[str, float],
    max_k: int,
    local_k: int,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Training-size conformal sweep",
                "",
                f"- Training sizes: 20, 40, … games (20-game chunks up to {max_k * 20} games, `--max-k` = max chunks).",
                "- Split scheme: by `GameId` with fixed partitions train_pool=60%, calibration=20%, test=20%. Forty games held out from train pool for hyperparameter tuning (tune-val); sweep uses remaining train games.",
                f"- Miscoverage level: alpha={alpha} (target coverage={1-alpha:.2f}).",
                f"- Label support: YardIndexClipped in [{MIN_IDX_Y}, {MAX_IDX_Y}] ({NUM_CLASSES} classes).",
                f"- Conformal method: interval-from-distribution central interval + Tibshirani-style locally weighted conformal padding q(x) using Gaussian kernel on predictive mean/std/entropy features (k={local_k}).",
                f"- Seed: {seed}.",
                f"- Frozen hyperparameters: Ridge (SGD) alpha={ridge_alpha} (tuned on CRPS); Tree (LightGBM multiclass, num_class={NUM_CLASSES}) params={tree_params} (tuned on CRPS); CNN full Zoo (CRPS loss, 20% val, early stopping patience=10, same as run_zoo).",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training-size conformal stability sweep.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=209, help="Seed for split and models (209 = day of first run).")
    parser.add_argument("--max-k", type=int, default=50, help="Sweep 20-game chunks up to max_k*20 games. Outputs sweep_{max_k}.csv/.png; one sweep_README.md.")
    parser.add_argument("--local-k", type=int, default=200, help="Calibration neighbors used for local bandwidth in Tibshirani-style weighted conformal intervals.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        alpha=args.alpha,
        seed=args.seed,
        out_dir=args.out_dir,
        processed_dir=args.processed_dir,
        raw_csv=args.raw_train_csv,
        max_k=args.max_k,
        local_k=args.local_k,
    )
