# da zoo

Replicating the zoo for uncertainty quantification.

## Setup

**Python 3.12 + venv**

```bash
# Install Python 3.12 if needed (Mac)
brew install python@3.12

# Create venv and install deps (from repo root)
./create_venv.sh
source .venv/bin/activate
```

Dependencies are in `requirements.txt` (includes `tensorflow-metal` for Mac GPU). The venv uses TensorFlow 2.18 for Metal compatibility.

## Workflow

1. **Prepare (once)**  
   Put raw data in `data/raw/train.csv`. Then:
   ```bash
   python prepare_data.py
   ```
   Writes `data/processed/train_x.npy`, `train_y.pkl`, `df_season.pkl`.

2. **Run zoo**  
   ```bash
   python run_zoo.py
   ```
   Loads processed data and runs 8-fold cross-validation (train 8 CNNs, report mean validation CRPS). After the first fold it prints an estimated total runtime. Use `CV_WORKERS = 1` for a single GPU (e.g. Mac Metal); set `CV_WORKERS = 2`–`8` to run folds in parallel on CPU.

## Original notebooks

Reference material lives in `original/`:

- [1st place zoo solution (v2)](original/1st_place_zoo_solution_v2.ipynb) — NFL Big Data Bowl 2020 winner
- [Pytorch version](original/pytorch_version.ipynb) — graph neural nets (torch geometric)
- [Player influence area CNN](original/my_solution.ipynb)
- [Graph convolutional network](original/nfl_graph_neural_networks_v1.ipynb)
