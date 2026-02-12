# Training-size conformal sweep

- Training sizes: 20, 40, … games (20-game chunks up to 100 games, `--max-k` = max chunks).
- Split scheme: by `GameId` with fixed partitions train_pool=60%, calibration=20%, test=20%. Forty games held out from train pool for hyperparameter tuning (tune-val); sweep uses remaining train games.
- Miscoverage level: alpha=0.1 (target coverage=0.90).
- Label support: YardIndexClipped in [71, 150] (80 classes).
- Conformal method: interval-from-distribution central interval + Tibshirani-style locally weighted conformal padding q(x) using Gaussian kernel on predictive mean/std/entropy features (k=200).
- Seed: 209.
- Frozen hyperparameters: Ridge (SGD) alpha=0.3333333333333333 (tuned on CRPS); Tree (LightGBM multiclass, num_class=80) params={'learning_rate': 0.05, 'max_depth': 5, 'min_child_samples': 50, 'reg_alpha': 0.5, 'reg_lambda': 0.5} (tuned on CRPS); CNN full Zoo (CRPS loss, 20% val, early stopping patience=10, same as run_zoo).