# =============================================================================
# AATIP — utils/metrics.py  |  Evaluation functions — temporal CV only
# =============================================================================
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    mean_absolute_percentage_error, mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

VC = CFG["validation"]


# ---------------------------------------------------------------------------
# TEMPORAL CROSS-VALIDATION — expanding window
# ---------------------------------------------------------------------------
def temporal_cv(model, X: pd.DataFrame, y: pd.Series,
                n_splits: int = None, scoring: str = "deficit_recall") -> dict:
    """
    Expanding-window temporal CV. Never mixes future into training.
    scoring options: 'deficit_recall', 'f1_macro'
    """
    n_splits = n_splits or CFG["temporal"]["cv_n_splits"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]

        if len(Xtr) < 5 or len(ytr.unique()) < 2:
            continue

        model.fit(Xtr, ytr)
        pred = model.predict(Xval)

        if scoring == "deficit_recall":
            classes = sorted(y.unique())
            if -1 in classes:
                _, recall, _, _ = precision_recall_fscore_support(
                    yval, pred, labels=classes, zero_division=0
                )
                score = recall[list(classes).index(-1)]
            else:
                score = f1_score(yval, pred, average="macro", zero_division=0)
        else:
            score = f1_score(yval, pred, average="macro", zero_division=0)

        scores.append(float(score))

    if not scores:
        return {"mean": 0.0, "std": 0.0, "folds": [], "scoring": scoring}

    return {
        "mean":    round(float(np.mean(scores)), 4),
        "std":     round(float(np.std(scores)), 4),
        "folds":   [round(s, 4) for s in scores],
        "scoring": scoring,
    }


# ---------------------------------------------------------------------------
# CLASSIFICATION SUMMARY
# ---------------------------------------------------------------------------
def classification_report(y_true, y_pred, context: str = "") -> dict:
    """
    Returns per-class precision/recall/F1 with priority on deficit recall.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    label_names = {-1: "deficit", 0: "neutral", 1: "surplus"}

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    result = {
        "context": context,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "classes": {}
    }
    for i, lbl in enumerate(labels):
        name = label_names.get(lbl, str(lbl))
        result["classes"][name] = {
            "label_value": int(lbl),
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]), 4),
            "f1":        round(float(f1[i]), 4),
            "support":   int(sup[i]),
        }

    deficit_recall = result["classes"].get("deficit", {}).get("recall", None)
    if deficit_recall is not None:
        thresh = VC["min_deficit_recall"]
        flag = "PASS" if deficit_recall >= thresh else "FAIL"
        print(f"  [{flag}] {context} | Deficit recall: {deficit_recall:.3f} "
              f"(threshold {thresh}) | Macro F1: {result['macro_f1']:.3f}")

    return result


# ---------------------------------------------------------------------------
# PRICE FORECAST METRICS
# ---------------------------------------------------------------------------
def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     context: str = "") -> dict:
    """MAPE, RMSE, and directional accuracy."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"mape": None, "rmse": None, "directional_acc": None, "n": int(mask.sum())}

    mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if len(y_true) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
    else:
        dir_acc = None

    thresh = VC["max_price_mape"]
    flag = "PASS" if mape <= thresh else "FAIL"
    dir_str = f"{dir_acc:.3f}" if dir_acc is not None else "N/A"
    print(f"  [{flag}] {context} | MAPE: {mape:.3f} (thr {thresh}) | "
          f"RMSE: {rmse:.5f} | DirAcc: {dir_str}")

    return {
        "mape": round(mape, 5),
        "rmse": round(rmse, 5),
        "directional_acc": round(dir_acc, 4) if dir_acc is not None else None,
        "n": int(mask.sum()),
        "context": context,
    }


# ---------------------------------------------------------------------------
# CORRIDOR RANK CORRELATION
# ---------------------------------------------------------------------------
def rank_correlation(scores: pd.Series, actuals: pd.Series,
                     context: str = "") -> dict:
    """Spearman rank correlation — measures ranking quality, not absolute fit."""
    mask = scores.notna() & actuals.notna()
    n = int(mask.sum())
    if n < 3:
        print(f"  [WARN] {context}: too few obs (n={n}) for rank correlation")
        return {"spearman_rho": None, "p_value": None, "n": n}

    rho, pval = spearmanr(scores[mask], actuals[mask])
    thresh = VC["min_rank_spearman"]
    flag = "PASS" if rho >= thresh else "FAIL"
    print(f"  [{flag}] {context} | Spearman ρ: {rho:.3f} (thr {thresh}) | p: {pval:.4f} | n: {n}")
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value":      round(float(pval), 6),
        "n":            n,
        "context":      context,
    }


# ---------------------------------------------------------------------------
# MIS LABEL — human-readable interpretation
# ---------------------------------------------------------------------------
def mis_label(mis_val: float) -> dict:
    """Returns label, severity, and recommended action for a MIS value."""
    if mis_val is None or not np.isfinite(mis_val):
        return {"label": "Unknown", "severity": "unknown", "color": "#999999"}
    t = CFG["mis"]
    if mis_val > t["crisis_threshold"]:
        return {"label": "Structural market failure",  "severity": "critical", "color": "#d62728"}
    if mis_val > t["strong_arbitrage"]:
        return {"label": "Strong arbitrage signal",    "severity": "high",     "color": "#e07b00"}
    if mis_val > t["arbitrage_threshold"]:
        return {"label": "Arbitrage opportunity",      "severity": "moderate", "color": "#f4a100"}
    if mis_val > t["convergence_threshold"]:
        return {"label": "Near convergence",           "severity": "low",      "color": "#2ca02c"}
    return     {"label": "Market converged",           "severity": "none",     "color": "#1f77b4"}
