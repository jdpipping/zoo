"""
Load prepared data and run the zoo: 8-fold CV training of CNNs.
Run prepare_data.py first. Expects data/processed/train_x.npy, train_y.pkl, df_season.pkl.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AvgPool2D, AvgPool1D,
    Input, BatchNormalization, Dense, Add, Lambda, Dropout, LayerNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.model_selection import KFold

DATA_DIR = "data/processed"
TRAIN_X_PATH = f"{DATA_DIR}/train_x.npy"
TRAIN_Y_PATH = f"{DATA_DIR}/train_y.pkl"
DF_SEASON_PATH = f"{DATA_DIR}/df_season.pkl"

MIN_IDX_Y, MAX_IDX_Y = 71, 150
NUM_CLASSES_Y = MAX_IDX_Y - MIN_IDX_Y + 1

CV_WORKERS = 1  # 1 = GPU (e.g. Mac); 2–8 = parallel on CPU


# GPU (Mac Metal)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", [g.name for g in gpus])
    except RuntimeError:
        pass
else:
    print("No GPU; using CPU. Mac: pip install tensorflow-metal")


def crps(y_true, y_pred):
    return K.mean(K.sum((K.cumsum(y_pred, axis=1) - K.cumsum(y_true, axis=1)) ** 2, axis=1)) / 199


def get_conv_net(num_classes_y):
    inp = Input(shape=(11, 10, 10), name="playersfeatures_input")
    X = Conv2D(128, (1, 1), activation="relu")(inp)
    X = Conv2D(160, (1, 1), activation="relu")(X)
    X = Conv2D(128, (1, 1), activation="relu")(X)
    Xmax = Lambda(lambda x: x * 0.3)(MaxPooling2D((1, 10))(X))
    Xavg = Lambda(lambda x: x * 0.7)(AvgPool2D((1, 10))(X))
    X = Add()([Xmax, Xavg])
    X = Lambda(lambda y: K.squeeze(y, 2))(X)
    X = BatchNormalization()(X)
    X = Conv1D(160, 1, activation="relu")(X)
    X = BatchNormalization()(X)
    X = Conv1D(96, 1, activation="relu")(X)
    X = BatchNormalization()(X)
    X = Conv1D(96, 1, activation="relu")(X)
    X = BatchNormalization()(X)
    Xmax = Lambda(lambda x: x * 0.3)(MaxPooling1D(11)(X))
    Xavg = Lambda(lambda x: x * 0.7)(AvgPool1D(11)(X))
    X = Add()([Xmax, Xavg])
    X = Lambda(lambda y: K.squeeze(y, 1))(X)
    X = Dense(96, activation="relu")(X)
    X = BatchNormalization()(X)
    X = Dense(256, activation="relu")(X)
    X = LayerNormalization()(X)
    X = Dropout(0.3)(X)
    out = Dense(num_classes_y, activation="softmax", name="output")(X)
    return Model(inputs=inp, outputs=out)


class Metric(Callback):
    """Val CRPS from cumulative predictions for EarlyStopping."""

    def __init__(self, model, callbacks, data):
        super().__init__()
        self._predict_model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for c in self.callbacks:
            c.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for c in self.callbacks:
            c.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_valid, y_valid = self.data[0], self.data[1]
        y_pred = self._predict_model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        logs["val_CRPS"] = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
        for c in self.callbacks:
            c.on_epoch_end(batch, logs)


def _train_one_fold(args):
    fold_idx, tdx, vdx, train_x, train_y, df_season, num_classes_y, min_idx_y = args
    t0 = time.perf_counter()
    X_train = train_x[tdx]
    X_val = train_x[vdx]
    y_train = train_y.iloc[tdx]["YardIndexClipped"].values
    y_val = train_y.iloc[vdx]["YardIndexClipped"].values
    season_val = df_season.iloc[vdx]["Season"].values

    y_train_oh = np.zeros((len(y_train), num_classes_y), np.int32)
    for irow, row in enumerate(y_train):
        y_train_oh[irow, row - min_idx_y] = 1
    y_val_oh = np.zeros((len(y_val), num_classes_y), np.int32)
    for irow, row in enumerate(y_val - min_idx_y):
        y_val_oh[irow, row] = 1

    val_idx = np.where(season_val != 2017)[0]
    X_val, y_val_oh = X_val[val_idx], y_val_oh[val_idx]
    y_train_oh = y_train_oh.astype("float32")
    y_val_oh = y_val_oh.astype("float32")

    model = get_conv_net(num_classes_y)
    es = EarlyStopping(monitor="val_CRPS", mode="min", restore_best_weights=True, verbose=0, patience=10)
    es.set_model(model)
    metric = Metric(model, [es], [X_val, y_val_oh])
    model.compile(loss=crps, optimizer=Adam(learning_rate=1e-3))
    model.fit(X_train, y_train_oh, epochs=50, batch_size=64, verbose=0, callbacks=[metric], validation_data=(X_val, y_val_oh))
    val_crps = min(model.history.history["val_CRPS"])
    return fold_idx, val_crps, (time.perf_counter() - t0) / 60.0


def estimate_cv_runtime(n_folds=8, minutes_per_fold=None):
    if minutes_per_fold is None:
        return None
    return {"minutes_per_fold": minutes_per_fold, "estimated_total_minutes": n_folds * minutes_per_fold}


def main():
    train_x = np.load(TRAIN_X_PATH)
    train_y = pd.read_pickle(TRAIN_Y_PATH)
    df_season = pd.read_pickle(DF_SEASON_PATH)

    models = []
    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    fold_splits = list(kf.split(train_x, train_y))
    score = []

    if CV_WORKERS <= 1:
        for i, (tdx, vdx) in enumerate(fold_splits):
            print(f"Fold {i}")
            t0 = time.perf_counter()
            X_train, X_val = train_x[tdx], train_x[vdx]
            y_train = train_y.iloc[tdx]["YardIndexClipped"].values
            y_val = train_y.iloc[vdx]["YardIndexClipped"].values
            season_val = df_season.iloc[vdx]["Season"].values

            y_train_oh = np.zeros((len(y_train), NUM_CLASSES_Y), np.int32)
            for irow, row in enumerate(y_train):
                y_train_oh[irow, row - MIN_IDX_Y] = 1
            y_val_oh = np.zeros((len(y_val), NUM_CLASSES_Y), np.int32)
            for irow, row in enumerate(y_val - MIN_IDX_Y):
                y_val_oh[irow, row] = 1

            val_idx = np.where(season_val != 2017)[0]
            X_val = X_val[val_idx]
            y_val_oh = y_val_oh[val_idx].astype("float32")
            y_train_oh = y_train_oh.astype("float32")

            model = get_conv_net(NUM_CLASSES_Y)
            es = EarlyStopping(monitor="val_CRPS", mode="min", restore_best_weights=True, verbose=0, patience=10)
            es.set_model(model)
            metric = Metric(model, [es], [X_val, y_val_oh])
            model.compile(loss=crps, optimizer=Adam(learning_rate=1e-3))
            model.fit(X_train, y_train_oh, epochs=50, batch_size=64, verbose=0, callbacks=[metric], validation_data=(X_val, y_val_oh))

            val_crps = min(model.history.history["val_CRPS"])
            elapsed = (time.perf_counter() - t0) / 60.0
            print(f"  val_CRPS={val_crps:.4f}  {elapsed:.1f} min")
            if i == 0:
                est = estimate_cv_runtime(8, elapsed)
                print(f"  -> Est. total 8 folds: ~{est['estimated_total_minutes']:.1f} min")
            score.append(val_crps)
            models.append(model)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        job_args = [
            (i, tdx, vdx, train_x, train_y, df_season, NUM_CLASSES_Y, MIN_IDX_Y)
            for i, (tdx, vdx) in enumerate(fold_splits)
        ]
        results = [None] * 8
        with ProcessPoolExecutor(max_workers=min(CV_WORKERS, 8)) as ex:
            futures = [ex.submit(_train_one_fold, a) for a in job_args]
            for fut in as_completed(futures):
                fold_idx, val_crps, elapsed = fut.result()
                results[fold_idx] = (val_crps, elapsed)
                print(f"Fold {fold_idx} done: val_CRPS={val_crps:.4f} ({elapsed:.1f} min)")
        score = [r[0] for r in results]
        avg_min = np.mean([r[1] for r in results])
        print(f"Avg fold {avg_min:.1f} min -> serial total ~{8 * avg_min:.1f} min")

    mean_crps = float(np.mean(score))
    print("Mean val CRPS:", mean_crps)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "zoo_cv_results.json"
    results_path.write_text(
        json.dumps(
            {
                "mean_val_crps": mean_crps,
                "fold_val_crps": [float(s) for s in score],
                "n_splits": 8,
                "random_state": 42,
                "epochs": 50,
                "batch_size": 64,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    print(f"Saved results to {results_path}")

    return models, score


def estimate_bootstrap_runtime(n_bootstrap, fit_minutes_per_model, n_data_fractions=6, n_model_families=3, parallel_workers=1, overhead_fraction=0.15):
    total_fits = n_bootstrap * n_data_fractions * n_model_families
    serial_min = total_fits * fit_minutes_per_model * (1 + overhead_fraction)
    return {"total_fits": total_fits, "wall_clock_hours": serial_min / 60.0 / max(parallel_workers, 1)}


def bootstrap_prediction_uncertainty(predictions_matrix):
    """[n_bootstrap, n_examples] -> variance and 90% interval width stats."""
    var_by_example = np.var(predictions_matrix, axis=0)
    q95 = np.quantile(predictions_matrix, 0.95, axis=0)
    q05 = np.quantile(predictions_matrix, 0.05, axis=0)
    width90 = q95 - q05
    return {"variance_mean": float(np.mean(var_by_example)), "width90_mean": float(np.mean(width90))}


if __name__ == "__main__":
    main()
