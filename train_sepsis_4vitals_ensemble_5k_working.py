#!/usr/bin/env python3
"""
train_sepsis_4vitals_ensemble_5k.py

Ensemble (LightGBM + LSTM) using only 4 vitals: HR, O2Sat, Temp, Resp.
Collects ~5000 windowed samples (balanced when possible) from all provided dataset folders (A+B),
trains base models and a small stacking logistic regression, and evaluates on a held-out test set.

Usage (example):
  python train_sepsis_4vitals_ensemble_5k.py --data_dirs "D:/new_Data/training_setA/training;D:/new_Data/training_setB/training_setB" \
      --out_dir "D:/models/ensemble_5k" --window 6 --target_samples 5000

Dependencies:
  pip install pandas numpy tqdm scikit-learn lightgbm matplotlib tensorflow joblib
"""
import os
import glob
import argparse
import json
import time
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import warnings
import math

# Keras / TF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import set_random_seed

warnings.filterwarnings("ignore", category=FutureWarning)

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)
tf.random.set_seed(SEED)

# constants
USE_FEATURES = ["HR", "O2Sat", "Temp", "Resp"]
TARGET_COL = "SepsisLabel"


# --- helpers for reading and features ---
def read_patient_file(path):
    """Robust read of a .psv patient file. Best-effort mapping."""
    try:
        df = pd.read_csv(path, sep="|", engine="python")
    except Exception:
        df = pd.read_table(path, sep="|", engine="python", header=None)
    # try to ensure columns exist
    canonical = [
        "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH",
        "PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
        "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI",
        "Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age","Gender","Unit1","Unit2","HospAdmTime",
        "ICULOS","SepsisLabel"
    ]
    if df.shape[1] == len(canonical):
        df.columns = canonical
    # Last column heuristic -> SepsisLabel
    if TARGET_COL not in df.columns:
        try:
            last_vals = pd.to_numeric(df.iloc[:, -1], errors="coerce")
            unique_vals = set(last_vals.dropna().unique())
            if unique_vals.issubset({0, 1}):
                cols = [f"col{i}" for i in range(df.shape[1]-1)] + ["SepsisLabel"]
                df.columns = cols
        except Exception:
            pass
    for c in USE_FEATURES + [TARGET_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def slope_of_series(arr):
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return 0.0
    x = np.arange(len(arr))[mask]
    y = arr[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return 0.0
    if not np.isfinite(m):
        return 0.0
    return float(m)


def window_stats(series, median_value):
    """Return dict of stats for a 1D array (length window_size). Fallback to median if all NaN."""
    series = np.asarray(series, dtype=float)
    mask = ~np.isnan(series)
    if mask.sum() == 0:
        return {
            "last": float(median_value),
            "mean": float(median_value),
            "std": 0.0,
            "min": float(median_value),
            "max": float(median_value),
            "delta": 0.0,
            "slope": 0.0,
            "misspct": 1.0
        }
    last = series[-1] if not np.isnan(series[-1]) else float(series[mask][-1])
    first = series[0] if not np.isnan(series[0]) else float(series[mask][0])
    return {
        "last": float(last),
        "mean": float(np.nanmean(series)),
        "std": float(np.nanstd(series)),
        "min": float(np.nanmin(series)),
        "max": float(np.nanmax(series)),
        "delta": float(last - first),
        "slope": float(slope_of_series(series)),
        "misspct": float(np.isnan(series).sum()) / float(len(series))
    }


def extract_windows_from_df(df, medians, window_size=6):
    """
    From one patient's df, extract windows (engineered features + raw windows + labels).
    Returns lists: engineered_rows (dicts), raw_windows (arrays), labels
    """
    if df.shape[0] < window_size:
        return [], [], []
    # forward/backfill to allow slopes to work better (but keep NaNs if entire col missing)
    df2 = df.copy()
    for f in USE_FEATURES:
        if f in df2.columns:
            df2[f] = df2[f].ffill().bfill()
        else:
            df2[f] = np.nan

    eng_rows = []
    raw_windows = []
    labels = []
    n = df2.shape[0]
    for i in range(window_size - 1, n):
        w = df2.iloc[i - window_size + 1: i + 1]
        row = {}
        raw = []
        for f in USE_FEATURES:
            series = w[f].values if f in w.columns else np.array([np.nan] * window_size)
            stats = window_stats(series, medians[f])
            row.update({f + "_last": stats["last"],
                        f + "_mean": stats["mean"],
                        f + "_std": stats["std"],
                        f + "_min": stats["min"],
                        f + "_max": stats["max"],
                        f + "_delta": stats["delta"],
                        f + "_slope": stats["slope"],
                        f + "_misspct": stats["misspct"]})
            raw.append([float(x) if not (x is None or (isinstance(x, float) and math.isnan(x))) else np.nan for x in series])
        # label at time i
        if TARGET_COL in df2.columns:
            lbl = df2[TARGET_COL].iloc[i]
            if pd.isna(lbl):
                continue
            eng_rows.append(row)
            raw_windows.append(np.array(raw).T)  # shape (window_size, n_features)
            labels.append(int(lbl))
    return eng_rows, raw_windows, labels


# --- model builders ---
def build_lstm(window_size, n_features, lstm_units=64, dropout=0.3):
    model = Sequential()
    model.add(Input(shape=(window_size, n_features)))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


# --- training wrapper for LightGBM compatibility ---
def train_lgbm_compatible(params, lgb_train, lgb_val, num_boost_round, early_stopping_rounds, log_interval=50):
    lgb_ver = getattr(lgb, "__version__", "0.0.0")
    major = 0
    try:
        major = int(lgb_ver.split(".")[0])
    except Exception:
        major = 0
    if early_stopping_rounds is None or int(early_stopping_rounds) <= 0:
        if major < 4:
            model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val],
                              valid_names=["train", "val"], num_boost_round=num_boost_round,
                              verbose_eval=log_interval)
        else:
            model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val],
                              valid_names=["train", "val"], num_boost_round=num_boost_round,
                              callbacks=[lgb.log_evaluation(log_interval)])
    else:
        if major < 4:
            model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val],
                              valid_names=["train", "val"], num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=log_interval)
        else:
            model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val],
                              valid_names=["train", "val"], num_boost_round=num_boost_round,
                              callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(log_interval)])
    return model


# --- plotting helpers ---
def save_confusion_matrix(y_true, y_pred, out_counts, out_norm, labels=["NoSepsis", "Sepsis"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Confusion matrix (counts)")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_counts)
    plt.close()

    with np.errstate(all="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5, 4))
    plt.imshow(cmn, interpolation="nearest", aspect="auto")
    plt.title("Confusion matrix (normalized)")
    plt.colorbar()
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            v = cmn[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_norm)
    plt.close()
    return cm


# ---------------- Main pipeline ----------------
def main(args):
    # parse directories (semicolon or comma separated)
    raw = args.data_dirs.strip()
    dirs = [d.strip() for d in raw.replace(",", ";").split(";") if d.strip()]
    if len(dirs) == 0:
        raise RuntimeError("No data directories provided.")

    # gather files
    all_files = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*.psv")))
        print(f"Found {len(found)} files in {d}")
        all_files.extend(found)
    all_files = sorted(list(set(all_files)))
    if len(all_files) == 0:
        raise RuntimeError("No psv files found.")
    print(f"Total files: {len(all_files)} (will sample up to {args.target_samples} windows).")

    # shuffle file list for random sampling
    random.shuffle(all_files)

    # compute medians on a small subset initially (fast)
    # use up to 2000 files or all if fewer
    sample_for_medians = min(2000, len(all_files))
    med_sample_files = all_files[:sample_for_medians]
    medians = {f: [] for f in USE_FEATURES}
    for p in tqdm(med_sample_files, desc="Gathering medians"):
        try:
            df = read_patient_file(p)
            for f in USE_FEATURES:
                if f in df.columns:
                    arr = df[f].dropna().values
                    if arr.size:
                        medians[f].extend(arr.tolist())
        except Exception:
            continue
    # finalize medians
    for f in USE_FEATURES:
        medians[f] = float(np.median(medians[f])) if len(medians[f]) else 0.0
    print("Medians:", medians)

    # sample windows until we have target_samples (balanced attempt)
    target = int(args.target_samples)
    half = target // 2
    pos_eng = []
    pos_raw = []
    neg_eng = []
    neg_raw = []
    files_used = 0

    for p in tqdm(all_files, desc="Scanning files to collect windows"):
        if len(pos_eng) >= half and len(neg_eng) >= (target - half):
            break
        try:
            df = read_patient_file(p)
            eng_rows, raw_windows, labels = extract_windows_from_df(df, medians, window_size=args.window)
            files_used += 1
            if len(labels) == 0:
                continue
            # iterate through windows, append to appropriate bins while respecting target counts
            for eng, raw, lbl in zip(eng_rows, raw_windows, labels):
                if lbl == 1:
                    if len(pos_eng) < half:
                        pos_eng.append(eng); pos_raw.append(raw)
                else:
                    if len(neg_eng) < (target - half):
                        neg_eng.append(eng); neg_raw.append(raw)
                if len(pos_eng) >= half and len(neg_eng) >= (target - half):
                    break
        except Exception:
            continue

    # if not enough positives found, relax and accept whatever we have
    n_pos = len(pos_eng)
    n_neg = len(neg_eng)
    total_collected = n_pos + n_neg
    if total_collected == 0:
        raise RuntimeError("No windowed samples collected. Check data files and window size.")

    print(f"Collected {total_collected} windows from {files_used} files (pos={n_pos}, neg={n_neg}).")

    # if not exactly target, combine and possibly undersample majority to reach target
    # create dataset lists
    X_eng_list = pos_eng + neg_eng
    X_raw_list = pos_raw + neg_raw
    y_list = [1] * n_pos + [0] * n_neg

    # shuffle dataset
    combined = list(zip(X_eng_list, X_raw_list, y_list))
    random.shuffle(combined)
    X_eng_list, X_raw_list, y_list = zip(*combined)
    X_eng_df = pd.DataFrame(X_eng_list)
    X_raw_arr = np.stack(X_raw_list, axis=0)  # shape (N, window, features)
    y_arr = np.array(y_list, dtype=int)

    # Final sample count
    N = len(y_arr)
    print("Final dataset size:", N, "; pos:", int((y_arr==1).sum()), "neg:", int((y_arr==0).sum()))

    # Split into train/val/test (60/20/20) stratified
    X_temp, X_test_df, raw_temp, raw_test_arr, y_temp, y_test = train_test_split(
        X_eng_df, X_raw_arr, y_arr, test_size=0.2, random_state=SEED, stratify=y_arr)
    X_train_df, X_val_df, raw_train_arr, raw_val_arr, y_train, y_val = train_test_split(
        X_temp, raw_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)  # 0.25*0.8 = 0.2

    print("Split sizes -> train:", len(y_train), "val:", len(y_val), "test:", len(y_test))

    # Convert to numpy for LightGBM (it handles NaNs)
    feature_cols = sorted(X_train_df.columns)
    X_train = X_train_df[feature_cols].values.astype(np.float32)
    X_val = X_val_df[feature_cols].values.astype(np.float32)
    X_test = X_test_df[feature_cols].values.astype(np.float32)

    # LSTM scaling: compute per-feature mean/std on raw_train_arr (over all timesteps & features)
    # raw arrays shape: (N, window, n_features)
    # We'll compute mean/std per feature across train set & timesteps
    n_features = raw_train_arr.shape[2]
    train_flat = raw_train_arr.reshape(-1, n_features)
    feat_mean = np.nanmean(train_flat, axis=0)
    feat_std = np.nanstd(train_flat, axis=0) + 1e-8
    # Fill NaNs in raw arrays with median per feature (from medians dict)
    # medians dict corresponds to USE_FEATURES order
    med_vector = np.array([medians[f] for f in USE_FEATURES], dtype=float)

    def preprocess_raw(raw_arr):
        # raw_arr shape (N, window, n_features)
        arr = np.array(raw_arr, dtype=float)
        # replace NaNs with medians first
        for fi in range(n_features):
            nan_mask = np.isnan(arr[:, :, fi])
            if nan_mask.any():
                arr[:, :, fi][nan_mask] = med_vector[fi]
        # normalize
        arr = (arr - feat_mean[None, None, :]) / feat_std[None, None, :]
        return arr

    X_raw_train = preprocess_raw(raw_train_arr)
    X_raw_val = preprocess_raw(raw_val_arr)
    X_raw_test = preprocess_raw(raw_test_arr)

    # ---------------- Train LightGBM ----------------
    print("Training LightGBM...")
    neg_count = float((y_train == 0).sum())
    pos_count = float((y_train == 1).sum())
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "seed": SEED,
        "scale_pos_weight": scale_pos_weight,
        "nthread": args.n_jobs
    }
    lgb_model = train_lgbm_compatible(lgb_params, lgb_train, lgb_val,
                                      num_boost_round=args.num_boost_round,
                                      early_stopping_rounds=args.early_stopping_rounds,
                                      log_interval=20)

    # LightGBM predictions
    lgb_val_prob = lgb_model.predict(X_val, num_iteration=getattr(lgb_model, "best_iteration", None) or None)
    lgb_test_prob = lgb_model.predict(X_test, num_iteration=getattr(lgb_model, "best_iteration", None) or None)
    lgb_val_auc = roc_auc_score(y_val, lgb_val_prob)
    lgb_test_auc = roc_auc_score(y_test, lgb_test_prob)
    print(f"LightGBM validation AUC: {lgb_val_auc:.4f}, test AUC: {lgb_test_auc:.4f}")

    # ---------------- Train LSTM ----------------
    print("Training LSTM...")
    window_size = X_raw_train.shape[1]
    n_feats = X_raw_train.shape[2]
    lstm_model = build_lstm(window_size, n_feats, lstm_units=args.lstm_units, dropout=args.lstm_dropout)
    # class weights for imbalance
    from sklearn.utils import class_weight
    classes_ = np.unique(y_train)
    cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes_, y=y_train)
    class_weight_dict = {int(classes_[i]): float(cw[i]) for i in range(len(classes_))}
    # callbacks
    os.makedirs(args.out_dir, exist_ok=True)
    lstm_path = os.path.join(args.out_dir, "lstm_model.keras")
    es = EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(lstm_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1)
    history = lstm_model.fit(X_raw_train, y_train,
                             validation_data=(X_raw_val, y_val),
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             class_weight=class_weight_dict,
                             callbacks=[es, mc],
                             verbose=2)
    # load best
    try:
        lstm_model = tf.keras.models.load_model(lstm_path)
    except Exception:
        pass
    lstm_val_prob = lstm_model.predict(X_raw_val, batch_size=args.batch_size).ravel()
    lstm_test_prob = lstm_model.predict(X_raw_test, batch_size=args.batch_size).ravel()
    lstm_val_auc = roc_auc_score(y_val, lstm_val_prob)
    lstm_test_auc = roc_auc_score(y_test, lstm_test_prob)
    print(f"LSTM validation AUC: {lstm_val_auc:.4f}, test AUC: {lstm_test_auc:.4f}")

    # ---------------- Stacking (logistic on val) ----------------
    print("Stacking predictions with logistic regression...")
    stack_X_val = np.vstack([lgb_val_prob, lstm_val_prob]).T
    stack_X_test = np.vstack([lgb_test_prob, lstm_test_prob]).T
    meta = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=SEED)
    meta.fit(stack_X_val, y_val)
    stack_test_prob = meta.predict_proba(stack_X_test)[:, 1]
    stack_test_auc = roc_auc_score(y_test, stack_test_prob)
    print(f"Stacked test AUC: {stack_test_auc:.4f}")

    # choose thresholds: use 0.5 or optimize for F1? We'll stick to 0.5 but print metrics
    def eval_model(name, probs, y_true, out_prefix):
        preds = (probs >= args.threshold).astype(int)
        print(f"\n=== {name} ===")
        print("AUC:", roc_auc_score(y_true, probs))
        print(classification_report(y_true, preds, digits=4))
        cm = confusion_matrix(y_true, preds)
        print("Confusion matrix:\n", cm)
        # save ROC curve
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}"); plt.legend(loc="lower right")
        roc_path = os.path.join(args.out_dir, out_prefix + "_roc.png")
        plt.savefig(roc_path); plt.close()
        # save confusion matrix plots
        cm_counts = os.path.join(args.out_dir, out_prefix + "_cm_counts.png")
        cm_norm = os.path.join(args.out_dir, out_prefix + "_cm_norm.png")
        save_confusion_matrix(y_true, preds, cm_counts, cm_norm)
        print("Saved ROC and confusion matrix plots to:", roc_path, cm_counts, cm_norm)

    # Evaluate and save
    os.makedirs(args.out_dir, exist_ok=True)
    # LightGBM
    eval_model("LightGBM", lgb_test_prob, y_test, "lgb_test")
    # LSTM
    eval_model("LSTM", lstm_test_prob, y_test, "lstm_test")
    # Stacked
    eval_model("Stacked", stack_test_prob, y_test, "stacked_test")

    # save models and metadata
    print("Saving models and metadata...")
    lgb_model.save_model(os.path.join(args.out_dir, "lgb_model.txt"))
    lstm_model.save(os.path.join(args.out_dir, "lstm_model.keras"), save_format="keras")
    joblib.dump(meta, os.path.join(args.out_dir, "meta_logreg.joblib"))
    # save scalers/medians/feature order
    meta_info = {
        "feature_cols": feature_cols,
        "medians": medians,
        "lstm_feature_mean": feat_mean.tolist(),
        "lstm_feature_std": feat_std.tolist(),
        "window": args.window,
        "target_samples": N,
        "params": {
            "lgb_params": lgb_params,
            "lstm_units": args.lstm_units,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
    }
    with open(os.path.join(args.out_dir, "preprocessing_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)
    print("Saved models and metadata to", args.out_dir)
    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble (LightGBM + LSTM) on 4 vitals using ~5k samples")
    parser.add_argument("--data_dirs", type=str, required=True,
                        help='Semicolon or comma separated directories, e.g. "D:/.../training;D:/.../training_setB"')
    parser.add_argument("--out_dir", type=str, default=r"D:/models/ensemble_5k", help="Directory to save models/plots")
    parser.add_argument("--window", type=int, default=6, help="Window size (timesteps)")
    parser.add_argument("--target_samples", type=int, default=5000, help="Approx number of windows to collect (total)")
    parser.add_argument("--val_frac", type=float, default=0.2, help="(Not used directly) final split is 60/20/20")
    parser.add_argument("--num_boost_round", type=int, default=500, help="LightGBM max boosting rounds")
    parser.add_argument("--early_stopping_rounds", type=int, default=50, help="Early stopping rounds for LightGBM")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for printing reports")
    parser.add_argument("--n_jobs", type=int, default=4, help="Threads for LightGBM")
    parser.add_argument("--lstm_units", type=int, default=64, help="Units in LSTM")
    parser.add_argument("--lstm_dropout", type=float, default=0.3, help="Dropout in LSTM")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for LSTM (with early stopping)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for LSTM")
    args = parser.parse_args()
    main(args)
