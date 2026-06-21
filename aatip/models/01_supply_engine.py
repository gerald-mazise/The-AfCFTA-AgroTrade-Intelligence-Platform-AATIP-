# =============================================================================
# AATIP — models/01_supply_engine.py  |  Supply & Deficit Classifier
# =============================================================================
# Classifies each country-month as surplus (1), neutral (0), or deficit (-1).
# Priority metric: DEFICIT RECALL — missing a deficit = food security failure.
#
# Models: RandomForest + GradientBoosting (sklearn only, no xgboost/lightgbm)
# Temporal CV: expanding window only — no random splits anywhere.
#
# INPUT:  AATIP_Intelligence_Master_144_Feature_Claude.csv
# OUTPUT: outputs/predictions/surplus_deficit_predictions.csv
#         outputs/model_artifacts/supply_engine_rf.pkl
#         outputs/model_artifacts/supply_engine_gb.pkl
#         governance/model_cards/01_supply_engine.json
# =============================================================================

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.features import SUPPLY_FEATURES, SUPPLY_TARGET, validate_registry
from utils.validators import (
    check_temporal_split, check_no_leakage, safe_features,
    impute_train_only, check_mis_range, check_corridor_completeness
)
from utils.metrics import temporal_cv, classification_report

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

PRED_DIR = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
ART_DIR  = os.path.join(BASE_DIR, CFG["data"]["artifacts_dir"])
GOV_DIR  = os.path.join(BASE_DIR, "governance", "model_cards")
for d in [PRED_DIR, ART_DIR, GOV_DIR]:
    os.makedirs(d, exist_ok=True)

SE = CFG["supply_engine"]


# ---------------------------------------------------------------------------
# 1. LOAD & RESHAPE  pairwise → country-month (exporter view)
# ---------------------------------------------------------------------------
def load_country_month(path: str) -> tuple:
    """
    The master CSV is pairwise (corridor-month).
    Each country appears as Exporter in multiple rows per month.
    Deduplicate to one row per (Country, Year, Month) — the country-month view.
    """
    print(f"\n[01] Loading: {os.path.basename(path)}")
    df = pd.read_csv(path)
    print(f"     Pairwise: {df.shape[0]} rows × {df.shape[1]} cols")

    # Validate
    validate_registry(df.columns.tolist(), verbose=False)
    check_corridor_completeness(df)
    check_mis_range(df, "full dataset")

    # Exporter view: take supply-related columns, rename Exporter → Country
    exp_cols = (
        ["Exporter", "Year", "Month"]
        + [f for f in SUPPLY_FEATURES + [SUPPLY_TARGET] if f in df.columns]
    )
    exp_df = (
        df[exp_cols]
        .rename(columns={"Exporter": "Country"})
        .drop_duplicates(subset=["Country", "Year", "Month"])
        .reset_index(drop=True)
    )
    print(f"     Country-month view: {exp_df.shape[0]} rows")
    print(f"     Countries: {sorted(exp_df['Country'].unique())}")
    return df, exp_df


# ---------------------------------------------------------------------------
# 2. TEMPORAL SPLIT & FEATURE PREP
# ---------------------------------------------------------------------------
def prepare(df: pd.DataFrame) -> tuple:
    """Temporal split, feature selection, train-only imputation."""
    train_years = CFG["temporal"]["train_years"]
    test_years  = CFG["temporal"]["test_years"]

    train = df[df["Year"].isin(train_years)].copy()
    test  = df[df["Year"].isin(test_years)].copy()

    # Drop rows where target is NaN (can't train or evaluate)
    train = train.dropna(subset=[SUPPLY_TARGET])
    test  = test.dropna(subset=[SUPPLY_TARGET])

    check_temporal_split(train, test, context="01_supply_engine")
    check_no_leakage(SUPPLY_FEATURES, context="01_supply_engine")

    # Use only features present in dataset
    features = safe_features(SUPPLY_FEATURES, df.columns.tolist(),
                              context="01_supply_engine")

    X_train, X_test, medians = impute_train_only(train, test, features)
    y_train = train[SUPPLY_TARGET].astype(int)
    y_test  = test[SUPPLY_TARGET].astype(int)

    # Class distribution
    dist = y_train.value_counts().to_dict()
    print(f"     Train class dist: {dist}  (target: deficit recall ≥ {CFG['validation']['min_deficit_recall']})")
    print(f"     Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(features)}")
    return X_train, X_test, y_train, y_test, features, medians, train, test


# ---------------------------------------------------------------------------
# 3A. RANDOM FOREST
# ---------------------------------------------------------------------------
def train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    print("\n[01] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=SE["rf_n_estimators"],
        max_depth=SE["rf_max_depth"],
        min_samples_leaf=SE["rf_min_samples_leaf"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = temporal_cv(rf, X_train, y_train, scoring="deficit_recall")
    print(f"     CV deficit recall: {cv['mean']:.3f} ± {cv['std']:.3f}  "
          f"folds: {cv['folds']}")
    rf.fit(X_train, y_train)
    return rf, cv


# ---------------------------------------------------------------------------
# 3B. GRADIENT BOOSTING
# ---------------------------------------------------------------------------
def train_gb(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    print("\n[01] Training Gradient Boosting...")

    # GradientBoosting needs labels remapped to 0,1,2 for multiclass
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map   = {0: -1, 1: 0, 2: 1}
    y_mapped  = y_train.map(label_map)

    gb = GradientBoostingClassifier(
        n_estimators=SE["gb_n_estimators"],
        max_depth=SE["gb_max_depth"],
        learning_rate=SE["gb_learning_rate"],
        subsample=0.8,
        random_state=42,
    )
    cv = temporal_cv(gb, X_train, y_mapped, scoring="f1_macro")
    print(f"     CV macro F1: {cv['mean']:.3f} ± {cv['std']:.3f}")
    gb.fit(X_train, y_mapped)
    return gb, label_map, inv_map, cv


# ---------------------------------------------------------------------------
# 4. EVALUATE ON TEST SET
# ---------------------------------------------------------------------------
def evaluate(rf, gb, inv_map, X_test, y_test, features) -> dict:
    print("\n[01] Test set evaluation (2022)...")

    rf_preds = rf.predict(X_test)
    rf_m = classification_report(y_test.values, rf_preds, "RandomForest_Test")

    gb_preds_raw = gb.predict(X_test)
    gb_preds = pd.Series(gb_preds_raw).map(inv_map).values
    gb_m = classification_report(y_test.values, gb_preds, "GradBoost_Test")

    # Feature importance from RF
    importance = pd.Series(
        rf.feature_importances_, index=features
    ).sort_values(ascending=False)
    print("\n     Top 10 features (RF importance):")
    for f, imp in importance.head(10).items():
        print(f"       {f:<45} {imp:.4f}")

    return {
        "rf": rf_m, "gb": gb_m,
        "rf_preds": rf_preds,
        "gb_preds": gb_preds,
        "feature_importance": importance.head(20).round(5).to_dict(),
    }


# ---------------------------------------------------------------------------
# 5. GENERATE PREDICTIONS ON FULL DATASET
# ---------------------------------------------------------------------------
def build_output(df_full: pd.DataFrame, rf, gb, inv_map,
                 features: list, medians: pd.Series) -> pd.DataFrame:
    print("\n[01] Generating predictions on full dataset...")

    exp_cols = (
        ["Exporter", "Year", "Month"]
        + [f for f in SUPPLY_FEATURES + [SUPPLY_TARGET] if f in df_full.columns]
    )
    out = (
        df_full[exp_cols]
        .rename(columns={"Exporter": "Country"})
        .drop_duplicates(subset=["Country", "Year", "Month"])
        .reset_index(drop=True)
    )

    X_all = out[features].fillna(medians[features])

    # RF predictions + probabilities
    out["Pred_Surplus_RF"] = rf.predict(X_all)
    proba = rf.predict_proba(X_all)
    for i, cls in enumerate(rf.classes_):
        lbl = {-1: "Deficit", 0: "Neutral", 1: "Surplus"}.get(cls, str(cls))
        out[f"P_{lbl}_RF"] = proba[:, i]

    # GB predictions
    gb_raw = gb.predict(X_all)
    out["Pred_Surplus_GB"] = pd.Series(gb_raw, index=out.index).map(inv_map).values

    # Consensus: RF is primary (balanced class weights + interpretable)
    out["Pred_Surplus_Score"] = out["Pred_Surplus_RF"]

    # Deficit flag for early warning (uses probability threshold)
    if "P_Deficit_RF" in out.columns:
        out["Deficit_Warning_Flag"] = (
            out["P_Deficit_RF"] >= SE["deficit_prob_threshold"]
        ).astype(int)

    path = os.path.join(PRED_DIR, "surplus_deficit_predictions.csv")
    out.to_csv(path, index=False)
    print(f"     Saved → {path}  ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# 6. SAVE ARTIFACTS & GOVERNANCE CARD
# ---------------------------------------------------------------------------
def save_artifacts(rf, gb, features, medians, metrics, cv_rf, cv_gb) -> None:
    with open(os.path.join(ART_DIR, "supply_engine_rf.pkl"), "wb") as f:
        pickle.dump({"model": rf, "features": features,
                     "medians": medians.to_dict()}, f)
    with open(os.path.join(ART_DIR, "supply_engine_gb.pkl"), "wb") as f:
        pickle.dump({"model": gb, "features": features,
                     "medians": medians.to_dict()}, f)

    card = {
        "engine":         "01_supply_engine",
        "models":         ["RandomForestClassifier", "GradientBoostingClassifier"],
        "target":         SUPPLY_TARGET,
        "classes":        {"-1": "deficit", "0": "neutral", "1": "surplus"},
        "primary_model":  "RandomForest (deficit recall optimised, class_weight=balanced)",
        "priority_metric": "deficit_recall",
        "train_years":    CFG["temporal"]["train_years"],
        "test_years":     CFG["temporal"]["test_years"],
        "n_features":     len(features),
        "features":       features,
        "cv_rf":          cv_rf,
        "cv_gb":          cv_gb,
        "test_metrics": {
            "rf_macro_f1":        metrics["rf"]["macro_f1"],
            "gb_macro_f1":        metrics["gb"]["macro_f1"],
            "rf_deficit_recall":  metrics["rf"]["classes"].get("deficit", {}).get("recall"),
            "gb_deficit_recall":  metrics["gb"]["classes"].get("deficit", {}).get("recall"),
        },
        "top_features": metrics["feature_importance"],
        "assumptions": [
            "class_weight=balanced: corrects for class imbalance without oversampling",
            "Imputation: column medians from train set applied to all data",
            "GB label remap: -1→0, 0→1, 1→2 (GradBoost requires non-negative int labels)",
        ],
        "caveats": [
            "Exporter_Surplus_Score is a pre-engineered composite, not raw production data",
            "Climate features at country level — sub-national variation not captured",
        ],
    }
    card_path = os.path.join(GOV_DIR, "01_supply_engine.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    print(f"     Governance card → {card_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(data_path: str = None) -> pd.DataFrame:
    print("=" * 70)
    print("  AATIP | Engine 01: Supply & Deficit Classifier")
    print("=" * 70)

    if data_path is None:
        data_path = os.path.join(BASE_DIR, CFG["data"]["master_csv"])

    df_full, df_country = load_country_month(data_path)

    X_train, X_test, y_train, y_test, features, medians, _, _ = (
        prepare(df_country)
    )

    rf,  cv_rf = train_rf(X_train, y_train)
    gb, label_map, inv_map, cv_gb = train_gb(X_train, y_train)

    metrics = evaluate(rf, gb, inv_map, X_test, y_test, features)
    predictions = build_output(df_full, rf, gb, inv_map, features, medians)
    save_artifacts(rf, gb, features, medians, metrics, cv_rf, cv_gb)

    print("\n[01] Complete ✓")
    print("=" * 70)
    return predictions


if __name__ == "__main__":
    run()
