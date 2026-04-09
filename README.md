# ML Perceptron Project (Extended)

This repository now includes:

- a single-layer perceptron from scratch,
- configurable training components (activation, loss, optimizer, regularization),
- explicit forward/backward training loop with mini-batches,
- multi-layer perceptron (MLP) from scratch,
- logic gate training (including XOR with MLP),
- financial-direction benchmark with hyperparameter tuning and baseline comparison.

## What Was Implemented

### 1) Single-Layer Perceptron (`perceptron_from_scratch.py`)

- Custom class with:
  - `fit`, `predict_proba`, `predict`, `score`, `evaluate`
- Configurable options:
  - Activations: `step`, `sigmoid`, `tanh`
  - Losses: `perceptron`, `log_loss`, `mse`
  - Optimizers: `sgd`, `momentum`
  - Regularization: `none`, `l1`, `l2`
- Added:
  - explicit forward pass
  - explicit backprop-style gradient updates
  - mini-batch training loop
  - loss and train-accuracy history tracking

### 2) Multi-Layer Perceptron (MLP) From Scratch

- Class: `MultiLayerPerceptronFromScratch`
- Supports:
  - hidden layers (custom sizes)
  - hidden activations (`tanh`, `relu`, `sigmoid`)
  - output sigmoid for binary classification
  - full backpropagation across layers
  - SGD and momentum updates
  - optional L1/L2 regularization

### 3) Logic Gates Training

- Single-layer perceptron:
  - learns `AND`, `OR`, `NAND`, `NOR`
  - fails on `XOR` (expected)
- MLP:
  - learns all gates including `XOR`

### 4) Financial Benchmark (`financial_model_benchmark.py`)

- Feature engineering from OHLCV:
  - returns, momentum, ranges, moving-average ratios, volume/volatility stats
- Time-aware split:
  - train/validation/test by chronology (no random leakage)
- Hyperparameter tuning:
  - custom MLP
  - baseline models: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting
- Threshold tuning:
  - optimized for balanced accuracy on validation
- Metrics:
  - accuracy, balanced accuracy, precision, recall, F1, log loss, ROC-AUC
- Efficiency improvements:
  - vectorized rolling standard deviation
  - reduced redundant retraining (best config per model family)
  - improved SVM cache usage

## Project Structure

- `Perceptron_Algorithm_Implementation.ipynb`: original notebook from repository
- `perceptron_from_scratch.py`: single-layer + MLP implementations and demos
- `financial_model_benchmark.py`: financial tuning and model comparison pipeline
- `requirements.txt`: original dependency file

## How To Run

### A) Run from-scratch perceptron + MLP demos

```bash
python3 perceptron_from_scratch.py
```

### B) Run financial benchmark (auto source selection)

```bash
python3 financial_model_benchmark.py
```

If online download is unavailable, it will automatically fall back to synthetic financial-like data.

### C) Run financial benchmark with your own local CSV

```bash
python3 financial_model_benchmark.py \
  --csv_path "/absolute/path/to/your_prices.csv" \
  --date_col Date \
  --open_col Open \
  --high_col High \
  --low_col Low \
  --close_col Close \
  --volume_col Volume
```

You can map to lowercase/custom column names by changing the CLI flags.

## Notes

- Financial directional prediction is a difficult task; moderate metrics can still be realistic.
- Prefer chronological splits and out-of-sample evaluation when validating trading models.




