import csv
from urllib.request import urlopen
import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from perceptron_from_scratch import MultiLayerPerceptronFromScratch


def load_stooq_ohlcv(ticker="AAPL.US"):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    with urlopen(url, timeout=15) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(content.splitlines())
    rows = [row for row in reader if row.get("Close") not in (None, "", "null")]
    if len(rows) < 300:
        raise ValueError(f"Not enough rows downloaded for {ticker}.")
    rows = sorted(rows, key=lambda r: r["Date"])
    return rows


def load_local_ohlcv_csv(
    csv_path,
    date_col="Date",
    open_col="Open",
    high_col="High",
    low_col="Low",
    close_col="Close",
    volume_col="Volume",
):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw.get(close_col) in (None, "", "null", "NaN"):
                continue
            rows.append(
                {
                    "Date": raw.get(date_col, ""),
                    "Open": raw.get(open_col, ""),
                    "High": raw.get(high_col, ""),
                    "Low": raw.get(low_col, ""),
                    "Close": raw.get(close_col, ""),
                    "Volume": raw.get(volume_col, "0"),
                }
            )
    if len(rows) < 300:
        raise ValueError(
            f"Local CSV has too few valid rows ({len(rows)}). Need at least ~300."
        )
    rows = sorted(rows, key=lambda r: r["Date"])
    return rows


def generate_synthetic_ohlcv(n=3200, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.00045, 0.0115, size=n)
    regime = np.where(np.arange(n) % 260 < 130, 0.00035, -0.00015)
    returns = base + regime
    price = 100 * np.cumprod(1 + returns)

    rows = []
    for i in range(n):
        c = price[i]
        o = c / (1 + rng.normal(0, 0.003))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0022)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0022)))
        v = 1_000_000 + 250_000 * abs(rng.normal())
        rows.append(
            {
                "Date": f"d{i}",
                "Open": str(o),
                "High": str(h),
                "Low": str(l),
                "Close": str(c),
                "Volume": str(v),
            }
        )
    return rows


def rolling_mean(x, window):
    out = np.full_like(x, np.nan, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def rolling_std(x, window):
    out = np.full_like(x, np.nan, dtype=float)
    x = x.astype(float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    csum2 = np.cumsum(np.insert(x * x, 0, 0.0))
    sums = csum[window:] - csum[:-window]
    sums2 = csum2[window:] - csum2[:-window]
    means = sums / window
    # var = E[x^2] - E[x]^2
    vars_ = np.maximum((sums2 / window) - (means * means), 0.0)
    stds = np.sqrt(vars_)
    out[window - 1 :] = stds
    return out


def build_financial_features(rows):
    close = np.array([float(r["Close"]) for r in rows], dtype=float)
    open_ = np.array([float(r["Open"]) for r in rows], dtype=float)
    high = np.array([float(r["High"]) for r in rows], dtype=float)
    low = np.array([float(r["Low"]) for r in rows], dtype=float)
    volume = np.array([float(r["Volume"]) for r in rows], dtype=float)

    ret_1 = (close[1:] - close[:-1]) / (close[:-1] + 1e-12)
    ret_1_aligned = np.full_like(close, np.nan, dtype=float)
    ret_1_aligned[1:] = ret_1

    ret_5 = np.full_like(close, np.nan, dtype=float)
    ret_10 = np.full_like(close, np.nan, dtype=float)
    ret_20 = np.full_like(close, np.nan, dtype=float)
    ret_5[5:] = (close[5:] - close[:-5]) / (close[:-5] + 1e-12)
    ret_10[10:] = (close[10:] - close[:-10]) / (close[:-10] + 1e-12)
    ret_20[20:] = (close[20:] - close[:-20]) / (close[:-20] + 1e-12)

    intraday = (close - open_) / (open_ + 1e-12)
    hl_range = (high - low) / (close + 1e-12)

    ma5 = rolling_mean(close, 5)
    ma20 = rolling_mean(close, 20)
    ma50 = rolling_mean(close, 50)
    ma_ratio_short = ma5 / (ma20 + 1e-12) - 1.0
    ma_ratio_long = ma20 / (ma50 + 1e-12) - 1.0

    log_vol = np.log(volume + 1.0)
    vol_ma10 = rolling_mean(log_vol, 10)
    vol_ma20 = rolling_mean(log_vol, 20)
    vol_dev10 = log_vol - vol_ma10
    vol_dev20 = log_vol - vol_ma20

    vol_std10 = rolling_std(ret_1, 10)
    vol_std20 = rolling_std(ret_1, 20)
    vol_std10_aligned = np.full_like(close, np.nan, dtype=float)
    vol_std20_aligned = np.full_like(close, np.nan, dtype=float)
    vol_std10_aligned[1:] = vol_std10
    vol_std20_aligned[1:] = vol_std20

    features = np.column_stack(
        [
            ret_1_aligned,
            ret_5,
            ret_10,
            ret_20,
            intraday,
            hl_range,
            close / (ma5 + 1e-12) - 1.0,
            ma_ratio_short,
            ma_ratio_long,
            vol_std10_aligned,
            vol_std20_aligned,
            vol_dev10,
            vol_dev20,
        ]
    )

    # Label: next-day direction
    y = (close[1:] > close[:-1]).astype(int)
    X = features[:-1]

    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]
    return X, y


def split_time_series(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.35, 0.65, 31)
    best_t = 0.5
    best_bal_acc = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t = t
    return best_t, best_bal_acc


def evaluate_predictions(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        eps = 1e-12
        out["log_loss"] = log_loss(y_true, np.clip(y_proba, eps, 1 - eps))
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        out["log_loss"] = np.nan
        out["roc_auc"] = np.nan
    return out


def tune_custom_mlp(X_train, y_train, X_val, y_val):
    configs = [
        {"hidden_layers": (8, 4), "learning_rate": 0.03, "optimizer": "momentum", "regularization": "l2", "reg_lambda": 0.0005},
        {"hidden_layers": (16, 8), "learning_rate": 0.02, "optimizer": "momentum", "regularization": "l2", "reg_lambda": 0.0010},
        {"hidden_layers": (16, 16), "learning_rate": 0.01, "optimizer": "momentum", "regularization": "l2", "reg_lambda": 0.0010},
        {"hidden_layers": (12, 6), "learning_rate": 0.02, "optimizer": "sgd", "regularization": "none", "reg_lambda": 0.0},
    ]

    best = None
    best_score = -1.0
    for cfg in configs:
        model = MultiLayerPerceptronFromScratch(
            input_dim=X_train.shape[1],
            hidden_layers=cfg["hidden_layers"],
            learning_rate=cfg["learning_rate"],
            n_epochs=1200,
            batch_size=64,
            hidden_activation="tanh",
            optimizer=cfg["optimizer"],
            momentum=0.9,
            regularization=cfg["regularization"],
            reg_lambda=cfg["reg_lambda"],
            shuffle=False,
            verbose=False,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)
        t, val_bal_acc = find_best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        # prioritize AUC, then balanced accuracy
        score = val_auc + 0.01 * val_bal_acc
        if score > best_score:
            best_score = score
            best = (model, cfg, t, val_bal_acc, val_auc)
    return best


def tune_sklearn_models(X_train_scaled, y_train, X_val_scaled, y_val, X_train_raw, X_val_raw):
    candidates = []

    # Logistic Regression (scaled)
    for c in [0.1, 1.0, 5.0]:
        model = LogisticRegression(C=c, max_iter=2000, random_state=42)
        model.fit(X_train_scaled, y_train)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        t, val_bal_acc = find_best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        candidates.append(
            ("LogisticRegression", model, "scaled", {"C": c}, t, val_bal_acc, val_auc)
        )

    # SVM (scaled)
    for c in [0.5, 1.0, 2.0]:
        model = SVC(C=c, kernel="rbf", probability=True, random_state=42, cache_size=500)
        model.fit(X_train_scaled, y_train)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        t, val_bal_acc = find_best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        candidates.append(("SVM_RBF", model, "scaled", {"C": c}, t, val_bal_acc, val_auc))

    # Random Forest (raw)
    for n_est, max_depth in [(200, 5), (300, 6), (300, 8)]:
        model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_raw, y_train)
        val_proba = model.predict_proba(X_val_raw)[:, 1]
        t, val_bal_acc = find_best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        candidates.append(
            (
                "RandomForest",
                model,
                "raw",
                {"n_estimators": n_est, "max_depth": max_depth},
                t,
                val_bal_acc,
                val_auc,
            )
        )

    # Gradient Boosting (raw)
    for n_est, lr, depth in [(150, 0.05, 2), (200, 0.05, 2), (200, 0.03, 3)]:
        model = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=depth,
            random_state=42,
        )
        model.fit(X_train_raw, y_train)
        val_proba = model.predict_proba(X_val_raw)[:, 1]
        t, val_bal_acc = find_best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        candidates.append(
            (
                "GradientBoosting",
                model,
                "raw",
                {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                t,
                val_bal_acc,
                val_auc,
            )
        )

    candidates.sort(key=lambda x: (x[-1], x[-2]), reverse=True)
    return candidates


def keep_best_config_per_model(candidates):
    best = {}
    for row in candidates:
        name = row[0]
        if name not in best:
            best[name] = row
    return list(best.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune and compare custom MLP vs baseline models on financial direction prediction."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL.US",
        help="Ticker for Stooq download (used only when --csv_path is not provided).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to local OHLCV CSV file. If provided, this is used instead of online download.",
    )
    parser.add_argument("--date_col", type=str, default="Date")
    parser.add_argument("--open_col", type=str, default="Open")
    parser.add_argument("--high_col", type=str, default="High")
    parser.add_argument("--low_col", type=str, default="Low")
    parser.add_argument("--close_col", type=str, default="Close")
    parser.add_argument("--volume_col", type=str, default="Volume")
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker
    print("=== Financial Models Benchmark ===")
    print(f"Ticker: {ticker}")

    if args.csv_path:
        rows = load_local_ohlcv_csv(
            csv_path=args.csv_path,
            date_col=args.date_col,
            open_col=args.open_col,
            high_col=args.high_col,
            low_col=args.low_col,
            close_col=args.close_col,
            volume_col=args.volume_col,
        )
        source = f"local_csv:{args.csv_path}"
    else:
        try:
            rows = load_stooq_ohlcv(ticker=ticker)
            source = "stooq"
        except Exception as exc:
            print(f"Live download unavailable ({exc}). Using synthetic financial-like data.")
            rows = generate_synthetic_ohlcv(n=3200, seed=42)
            source = "synthetic"

    X, y = build_financial_features(rows)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_time_series(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # ---- Tune custom MLP ----
    mlp_model, mlp_cfg, mlp_t, mlp_val_bal_acc, mlp_val_auc = tune_custom_mlp(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    # retrain MLP on train+val
    X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])
    best_mlp = MultiLayerPerceptronFromScratch(
        input_dim=X_trainval_scaled.shape[1],
        hidden_layers=mlp_cfg["hidden_layers"],
        learning_rate=mlp_cfg["learning_rate"],
        n_epochs=1500,
        batch_size=64,
        hidden_activation="tanh",
        optimizer=mlp_cfg["optimizer"],
        momentum=0.9,
        regularization=mlp_cfg["regularization"],
        reg_lambda=mlp_cfg["reg_lambda"],
        shuffle=False,
        verbose=False,
        random_state=42,
    )
    best_mlp.fit(X_trainval_scaled, y_trainval)
    mlp_test_proba = best_mlp.predict_proba(X_test_scaled)
    mlp_test_pred = (mlp_test_proba >= mlp_t).astype(int)
    mlp_metrics = evaluate_predictions(y_test, mlp_test_pred, mlp_test_proba)

    # ---- Tune sklearn baselines ----
    candidates = tune_sklearn_models(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_train_raw,
        X_val_raw,
    )
    candidates = keep_best_config_per_model(candidates)

    # retrain each best candidate on train+val and evaluate on test
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])
    results = []

    # Add custom MLP first
    results.append(
        {
            "model": "CustomMLP",
            "val_bal_acc": mlp_val_bal_acc,
            "val_auc": mlp_val_auc,
            "threshold": mlp_t,
            "config": mlp_cfg,
            **mlp_metrics,
        }
    )

    # Use one best config per baseline family for cleaner/faster comparison
    for name, _, data_kind, cfg, threshold, val_bal_acc, val_auc in candidates:
        if name == "LogisticRegression":
            model = LogisticRegression(C=cfg["C"], max_iter=2000, random_state=42)
            model.fit(X_trainval_scaled, y_trainval)
            test_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif name == "SVM_RBF":
            model = SVC(C=cfg["C"], kernel="rbf", probability=True, random_state=42)
            model.fit(X_trainval_scaled, y_trainval)
            test_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_trainval_raw, y_trainval)
            test_proba = model.predict_proba(X_test_raw)[:, 1]
        else:
            model = GradientBoostingClassifier(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                random_state=42,
            )
            model.fit(X_trainval_raw, y_trainval)
            test_proba = model.predict_proba(X_test_raw)[:, 1]

        test_pred = (test_proba >= threshold).astype(int)
        metrics = evaluate_predictions(y_test, test_pred, test_proba)
        results.append(
            {
                "model": name,
                "val_bal_acc": val_bal_acc,
                "val_auc": val_auc,
                "threshold": threshold,
                "config": cfg,
                **metrics,
            }
        )

    results.sort(key=lambda row: (row["accuracy"], row["balanced_accuracy"]), reverse=True)

    print(f"\nData source: {source}")
    print(
        f"Samples -> train:{len(y_train)} val:{len(y_val)} test:{len(y_test)} "
        f"| positive rate test:{np.mean(y_test):.3f}"
    )
    print("\n=== Ranked Test Results ===")
    header = (
        f"{'model':<18} {'val_bal':>8} {'val_auc':>8} {'test_acc':>9} {'bal_acc':>8} "
        f"{'precision':>10} {'recall':>8} {'f1':>8} {'log_loss':>10} {'thr':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['model']:<18} {row['val_bal_acc']:>8.4f} {row['val_auc']:>8.4f} "
            f"{row['accuracy']:>9.4f} {row['balanced_accuracy']:>8.4f} "
            f"{row['precision']:>10.4f} {row['recall']:>8.4f} {row['f1']:>8.4f} "
            f"{row['log_loss']:>10.4f} {row['threshold']:>6.2f}"
        )

    best = results[0]
    print("\nBest model summary:")
    print(f"  Model      : {best['model']}")
    print(f"  Config     : {best['config']}")
    print(f"  Test Acc   : {best['accuracy']:.4f}")
    print(f"  Test F1    : {best['f1']:.4f}")
    print(f"  Test LogLoss: {best['log_loss']:.4f}")


if __name__ == "__main__":
    main()
