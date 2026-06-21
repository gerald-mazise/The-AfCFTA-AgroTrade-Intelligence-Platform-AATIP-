# =============================================================================
# AATIP — models/04_informal_trade_engine.py  |  Informal Trade Engine
# =============================================================================
# Reconstructs hidden trade flows that formal statistics cannot observe.
# When MIS > 0 and formal trade is suppressed, informal trade is absorbing
# the arbitrage — this engine quantifies that probability.
#
# Architecture: rule trigger → anomaly score → probability score → volume estimate
#
# P_informal = 0.40 × P_MIS + 0.30 × P_Persistence + 0.30 × P_Anomaly
#
# Calibrated against UNCTAD benchmark: informal ≈ 2–4× formal for EAC staples.
# ALL ESTIMATES ARE INFERRED — interpret as signals, not measurements.
#
# INPUT:  master CSV + 03 corridor rankings (optional)
# OUTPUT: outputs/predictions/informal_trade_estimates.csv
#         governance/model_cards/04_informal_trade_engine.json
# =============================================================================

import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.features import INFORMAL_PROBABILITY_FEATURES
from utils.validators import safe_features

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

PRED_DIR = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
GOV_DIR  = os.path.join(BASE_DIR, "governance", "model_cards")
for d in [PRED_DIR, GOV_DIR]:
    os.makedirs(d, exist_ok=True)

IT = CFG["informal_trade"]


# ---------------------------------------------------------------------------
# STEP 1 — RULE TRIGGER
# ---------------------------------------------------------------------------
def compute_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies months where the conditions for informal trade are structurally met:
    - MIS > 0: arbitrage opportunity exists
    - Formal_Trade_Intensity < threshold: formal channels are not absorbing it
    - Price_Convergence_Failure: markets are not correcting
    """
    df = df.copy()

    df["IT_Rule_Trigger"] = (
        (df["MIS"] > IT["mis_trigger"]) &
        (df["Formal_Trade_Intensity"].fillna(1) < IT["formal_intensity_max"])
    ).astype(int)

    df["IT_Persistence_Flag"] = (
        (df["MIS"] > IT["mis_trigger"]) &
        (df["MIS_Persistence_Months"].fillna(0) >= CFG["mis"]["persistence_months"])
    ).astype(int)

    n_t = df["IT_Rule_Trigger"].sum()
    n_p = df["IT_Persistence_Flag"].sum()
    print(f"     Rule triggers: {n_t}/{len(df)} ({n_t/len(df)*100:.1f}%)")
    print(f"     Persistence flags: {n_p}/{len(df)}")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — ANOMALY SCORE (Isolation Forest)
# ---------------------------------------------------------------------------
def compute_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest on corridor-level MIS/trade features.
    High anomaly score = price-trade decoupling = signature of informal flows.
    contamination=0.15: ~15% of months are anomalous (informal trade is common).
    """
    anomaly_features = [
        "MIS", "Price_Gap_USD_kg", "Formal_Trade_Intensity",
        "MIS_Zscore", "MIS_Persistence_Months",
    ]
    features = safe_features(anomaly_features, df.columns.tolist(),
                              context="anomaly_isolation_forest")

    df = df.copy()
    X = df[features].fillna(0).values

    iso = IsolationForest(
        n_estimators=100,
        contamination=IT["isolation_contamination"],
        random_state=42,
    )
    iso.fit(X)

    # Invert so higher = more anomalous (more likely informal)
    raw_scores = -iso.score_samples(X)
    scaler = MinMaxScaler()
    df["Anomaly_Score"] = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

    n_anomalous = (iso.predict(X) == -1).sum()
    print(f"     Anomalous months: {n_anomalous}/{len(df)} "
          f"(contamination={IT['isolation_contamination']})")
    return df


# ---------------------------------------------------------------------------
# STEP 3 — PROBABILISTIC SCORE
# ---------------------------------------------------------------------------
def compute_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-component probability:
      Component 1: MIS → sigmoid transform (high MIS = high informal probability)
      Component 2: Persistence → normalised months of sustained positive MIS
      Component 3: Anomaly → IsolationForest score (already [0,1])
    """
    df = df.copy()

    # Component 1: MIS via sigmoid (centred on 0, steepness=2)
    mis_z = df["MIS_Zscore"].fillna(0).clip(-4, 4)
    df["P_MIS"] = 1 / (1 + np.exp(-2 * mis_z))

    # Component 2: persistence (normalised against 12-month ceiling)
    pers_max = 12.0
    df["P_Persistence"] = (
        df["MIS_Persistence_Months"].fillna(0).clip(0, pers_max) / pers_max
    )

    # Component 3: anomaly (already in [0,1])
    df["P_Anomaly"] = df.get("Anomaly_Score", pd.Series(0.3, index=df.index))

    # Combined probability
    df["P_Informal"] = (
        IT["w_mis"]         * df["P_MIS"]
        + IT["w_persistence"] * df["P_Persistence"]
        + IT["w_anomaly"]     * df["P_Anomaly"]
    ).clip(0, 1)

    df["IT_High_Confidence"] = (df["P_Informal"] >= IT["high_confidence"]).astype(int)

    print(f"     P_Informal: mean={df['P_Informal'].mean():.3f}, "
          f"median={df['P_Informal'].median():.3f}")
    print(f"     High-confidence flags: {df['IT_High_Confidence'].sum()}")
    return df


# ---------------------------------------------------------------------------
# STEP 4 — VOLUME ESTIMATION
# Calibrated against UNCTAD: informal ≈ 2–4× formal
# ---------------------------------------------------------------------------
def estimate_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates implied informal trade volume as a range (low, mid, high).
    Formula: informal_vol = P_informal × formal_vol × UNCTAD_ratio
    Presented as a range because the UNCTAD calibration itself is a range.
    """
    df = df.copy()

    # Formal volume base
    formal_vol = df.get(
        "Exporter_Export_tonnes",
        pd.Series(5000.0, index=df.index)
    ).fillna(5000.0).clip(lower=1)

    lo = IT["unctad_ratio_low"]
    hi = IT["unctad_ratio_high"]

    df["IT_Vol_Low_tonnes"]  = (formal_vol * lo * df["P_Informal"]).round(0)
    df["IT_Vol_High_tonnes"] = (formal_vol * hi * df["P_Informal"]).round(0)
    df["IT_Vol_Mid_tonnes"]  = (
        (df["IT_Vol_Low_tonnes"] + df["IT_Vol_High_tonnes"]) / 2
    ).round(0)

    # Corridor-level intensity index (normalised within corridor)
    df["IT_Intensity"] = (
        df.groupby("Pair_ID")["P_Informal"]
        .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    )
    # Rolling 3M smoothed intensity
    df = df.sort_values(["Pair_ID", "Year", "Month"])
    df["IT_Intensity_MA3"] = (
        df.groupby("Pair_ID")["IT_Intensity"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    total = df["IT_Vol_Mid_tonnes"].sum()
    print(f"     Implied informal volume (mid estimate): {total:,.0f} tonnes total")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(data_path: str = None, rankings_path: str = None) -> pd.DataFrame:
    print("=" * 70)
    print("  AATIP | Engine 04: Informal Trade Engine")
    print("=" * 70)

    if data_path is None:
        data_path = os.path.join(BASE_DIR, CFG["data"]["master_csv"])

    df = pd.read_csv(data_path)
    print(f"\n[04] Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    if rankings_path and os.path.exists(rankings_path):
        rnk = pd.read_csv(rankings_path)
        merge_cols = [c for c in ["Pair_ID","Year","Month","Trade_Score_Final",
                                   "Gate_Pass","Active_Arbitrage"] if c in rnk.columns]
        df = df.merge(rnk[merge_cols], on=["Pair_ID","Year","Month"], how="left")
        print(f"     Engine 03 rankings merged")

    # Pipeline
    print()
    df = compute_triggers(df)
    df = compute_anomaly(df)
    df = compute_probability(df)
    df = estimate_volume(df)

    # Per-corridor summary
    print("\n     Corridor P_Informal summary:")
    summary = (
        df.groupby("Pair_ID")["P_Informal"]
        .agg(["mean", "max"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    for corridor, row in summary.iterrows():
        flag = " ← HIGH" if row["mean"] >= IT["high_confidence"] else ""
        print(f"     {corridor}: mean={row['mean']:.3f}, max={row['max']:.3f}{flag}")

    # Output
    out_cols = [
        "Pair_ID", "Exporter", "Importer", "Year", "Month",
        "MIS", "MIS_Persistence_Months",
        "IT_Rule_Trigger", "IT_Persistence_Flag",
        "P_MIS", "P_Persistence", "P_Anomaly",
        "P_Informal", "IT_High_Confidence",
        "Anomaly_Score",
        "IT_Vol_Low_tonnes", "IT_Vol_Mid_tonnes", "IT_Vol_High_tonnes",
        "IT_Intensity", "IT_Intensity_MA3",
        "Formal_Trade_Intensity", "Arbitrage_Profit_USD_kg",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    out = df[out_cols].copy()

    path = os.path.join(PRED_DIR, "informal_trade_estimates.csv")
    out.to_csv(path, index=False)
    print(f"\n     Saved → {path}  ({len(out)} rows)")

    # Governance card
    card = {
        "engine": "04_informal_trade_engine",
        "architecture": "rule_trigger → IsolationForest_anomaly → probability → volume",
        "probability_weights": {
            "P_MIS":         IT["w_mis"],
            "P_Persistence": IT["w_persistence"],
            "P_Anomaly":     IT["w_anomaly"],
        },
        "anomaly_detection": {
            "model": "IsolationForest",
            "contamination": IT["isolation_contamination"],
        },
        "unctad_calibration": {
            "source": "UNCTAD EAC staple grain benchmark",
            "ratio_range": f"{IT['unctad_ratio_low']}× – {IT['unctad_ratio_high']}× formal",
        },
        "caveats": [
            "ALL volume estimates are INFERRED — not measured, not observed",
            "P_Informal is a probabilistic signal, not a classification",
            "UNCTAD calibration applies to EAC staple grains specifically",
            "Anomaly score is relative to this dataset — not an absolute benchmark",
        ],
    }
    with open(os.path.join(GOV_DIR, "04_informal_trade_engine.json"), "w") as f:
        json.dump(card, f, indent=2)

    print("\n[04] Complete ✓")
    print("=" * 70)
    return out


if __name__ == "__main__":
    run()
