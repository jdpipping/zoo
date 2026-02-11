# Training-size conformal sweep

- Training sizes: 20, 40, … games (20-game chunks up to 100 games, `--max-k` = max chunks).
- Split scheme: by `GameId` with fixed partitions train_pool=60%, calibration=20%, test=20%.
- Miscoverage level: alpha=0.1 (target coverage=0.90).
- Label support: YardIndexClipped in [71, 150] (80 classes).
- Conformal method: interval-from-distribution central interval + calibration padding q via finite-sample conformal quantile.
- Seed: 123.
- Frozen hyperparameters: Ridge (SGD) alpha=0.3333333333333333; Tree (LightGBM multiclass, num_class=80) params={'learning_rate': 0.05, 'max_depth': 10}; CNN full Zoo (CRPS loss, 20% val, early stopping patience=10, same as run_zoo).