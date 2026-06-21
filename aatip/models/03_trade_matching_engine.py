# =============================================================================
# AATIP — models/03_trade_matching_engine.py  |  Trade Matching Engine
# =============================================================================
# Ranks corridors by trade opportunity using a STRICTLY ORDERED three-layer
# hierarchy. The order is non-negotiable:
#
#   Layer 1: Economic composite score  — theory-grounded floor
#   Layer 2: Lasso ML refinement       — optimises ranking against history
#   Layer 3: Hard rule gates           — feasibility filters, override all
#
# Layer 2 refines Layer 1. Layer 3 can zero out both.
# No layer can rescue a corridor that fails a hard gate.
#
# INPUT:  master CSV + 01 surplus predictions + 02 forward MIS (both optional)
# OUTPUT: outputs/predictions/corridor_rankings.csv
#         outputs/reports/lasso_coefficients.csv
#         governance/model_cards/03_trade_matching_engine.json
# =============================================================================

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.features import (
    TRADE_ECONOMIC_FEATURES, TRADE_LASSO_FEATURES,
    TRADE_LASSO_TARGET, HARD_GATE_FEATURES, validate_registry
)
from utils.validators import (
    check_temporal_split, check_no_leakage, safe_features, impute_train_only
)
from utils.metrics import rank_correlation

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

PRED_DIR = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
ART_DIR  = os.path.join(BASE_DIR, CFG["data"]["artifacts_dir"])
REP_DIR  = os.path.join(BASE_DIR, CFG["data"]["reports_dir"])
GOV_DIR  = os.path.join(BASE_DIR, "governance", "model_cards")
for d in [PRED_DIR, ART_DIR, REP_DIR, GOV_DIR]:
    os.makedirs(d, exist_ok=True)

TM = CFG["trade_matching"]
TRAIN_YEARS = CFG["temporal"]["train_years"]
TEST_YEARS  = CFG["temporal"]["test_years"]


# ---------------------------------------------------------------------------
# LAYER 1 — ECONOMIC COMPOSITE SCORE
# Theory-grounded. No fitting. Always computed. Always interpretable.
# ---------------------------------------------------------------------------
def compute_economic_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Trade_Score_Econ using World Bank SSA trade-cost elasticity weights.
    All components normalised to [0, 1] before weighting.

    Weights (from config):
      +0.35 × MIS_MA3          (primary opportunity signal)
      +0.30 × Supply_Confidence (exporter can deliver)
      +0.20 × Route_Feasibility (logistics are viable)
      -0.15 × Market_Friction   (costs subtracted)
    """
    df = df.copy()

    def norm01(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

    mis_col = "MIS_MA3" if "MIS_MA3" in df.columns else "MIS"

    mis_n   = norm01(df[mis_col].fillna(df["MIS"]))
    sup_n   = norm01(df["Supply_Confidence"].fillna(0))
    rte_n   = norm01(df["Route_Feasibility"].fillna(0))
    fri_n   = norm01(df["Market_Friction_Index"].fillna(0))

    raw = (
        TM["w_mis_ma3"]          * mis_n
        + TM["w_supply_confidence"] * sup_n
        + TM["w_route_feasibility"] * rte_n
        - TM["w_market_friction"]   * fri_n
    )
    df["Trade_Score_Econ"] = norm01(raw)

    print(f"     Economic score range: "
          f"{df['Trade_Score_Econ'].min():.3f} – {df['Trade_Score_Econ'].max():.3f}")
    print(f"     Weights: MIS={TM['w_mis_ma3']}, "
          f"Supply={TM['w_supply_confidence']}, "
          f"Route={TM['w_route_feasibility']}, "
          f"Friction=-{TM['w_market_friction']}")
    return df


# ---------------------------------------------------------------------------
# LAYER 2 — LASSO ML REFINEMENT
# ---------------------------------------------------------------------------
def fit_lasso(df: pd.DataFrame) -> tuple:
    """
    LassoCV on log(Exporter_Export_tonnes). Alpha selected via TimeSeriesSplit.
    Returns fitted pipeline, coefficient DataFrame, and train medians.
    """
    print("\n[03] Fitting Lasso refinement (Layer 2)...")
    check_no_leakage(TRADE_LASSO_FEATURES, context="03_lasso")
    features = safe_features(TRADE_LASSO_FEATURES, df.columns.tolist(),
                              context="lasso_features")

    # Drop rows where target is null
    df_model = df.dropna(subset=[TRADE_LASSO_TARGET]).copy()
    df_model["log_target"] = np.log1p(df_model[TRADE_LASSO_TARGET].clip(lower=0))

    train = df_model[df_model["Year"].isin(TRAIN_YEARS)]
    test  = df_model[df_model["Year"].isin(TEST_YEARS)]

    if len(train) < 20:
        print("     [WARN] Insufficient data for Lasso — using economic score only")
        return None, pd.DataFrame(), pd.Series()

    check_temporal_split(train, test, context="03_lasso")
    X_train, X_test, medians = impute_train_only(train, test, features)
    y_train = train["log_target"]
    y_test  = test["log_target"] if len(test) > 0 else pd.Series(dtype=float)

    tscv = TimeSeriesSplit(n_splits=CFG["temporal"]["cv_n_splits"])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  LassoCV(
            n_alphas=TM["lasso_n_alphas"],
            max_iter=TM["lasso_max_iter"],
            cv=tscv,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    alpha = pipe.named_steps["lasso"].alpha_
    print(f"     Best Lasso α (CV): {alpha:.6f}")

    # Coefficient table
    coefs = pipe.named_steps["lasso"].coef_
    coef_df = (
        pd.DataFrame({
            "Feature": features,
            "Coefficient": coefs,
            "Abs_Coef": np.abs(coefs),
        })
        .sort_values("Abs_Coef", ascending=False)
    )
    nonzero = coef_df[coef_df["Abs_Coef"] > 0]
    print(f"     Non-zero features: {len(nonzero)}/{len(features)}")
    print("     Top 10 Lasso coefficients:")
    for _, row in nonzero.head(10).iterrows():
        print(f"       {row['Feature']:<45} {row['Coefficient']:+.5f}")

    # Validate ranking quality on test set
    if len(X_test) > 0 and len(y_test) > 0:
        ml_preds = pipe.predict(X_test)
        rank_correlation(pd.Series(ml_preds), y_test, context="Lasso_TestSet")

    # Save coefficients
    coef_path = os.path.join(REP_DIR, "lasso_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"     Coefficients saved → {coef_path}")

    return pipe, coef_df, medians


# ---------------------------------------------------------------------------
# LAYER 3 — HARD RULE GATES
# These are not model scores. They are binary feasibility constraints.
# A corridor failing a gate gets Trade_Score_Final = 0 regardless of layers 1–2.
# ---------------------------------------------------------------------------
def apply_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    min_route    = TM["min_route_feasibility"]
    max_friction = TM["max_market_friction"]

    df["Gate_Route"]   = (df["Route_Feasibility"].fillna(0) >= min_route).astype(int)
    df["Gate_Friction"] = (df["Market_Friction_Index"].fillna(1) <= max_friction).astype(int)
    df["Gate_Pass"]    = (df["Gate_Route"] & df["Gate_Friction"]).astype(int)

    n_blocked = (df["Gate_Pass"] == 0).sum()
    print(f"\n[03] Hard gates: {n_blocked}/{len(df)} observations blocked "
          f"({n_blocked/len(df)*100:.1f}%)")
    print(f"     Route threshold: ≥{min_route}  |  Friction threshold: ≤{max_friction}")
    return df


# ---------------------------------------------------------------------------
# COMBINE ALL LAYERS → FINAL RANKING
# ---------------------------------------------------------------------------
def rank_corridors(df: pd.DataFrame,
                   lasso_pipe, lasso_medians) -> pd.DataFrame:
    """
    Produces final corridor ranking per (Year, Month).
    Combined score = 0.6 × Econ + 0.4 × ML (where ML is available)
    Final score zeroed for gate failures.
    """
    features = safe_features(TRADE_LASSO_FEATURES, df.columns.tolist())

    # Layer 1
    df = compute_economic_score(df)

    # Layer 2 (ML)
    if lasso_pipe is not None and len(features) > 0:
        X_all = df[features].fillna(lasso_medians[features])
        ml_raw = lasso_pipe.predict(X_all)
        lo, hi = ml_raw.min(), ml_raw.max()
        df["Trade_Score_ML"] = (
            pd.Series((ml_raw - lo) / (hi - lo + 1e-9), index=df.index)
        )
    else:
        df["Trade_Score_ML"] = df["Trade_Score_Econ"]

    # Combined: 60% economic (theory) + 40% ML (empirical)
    df["Trade_Score_Combined"] = (
        0.60 * df["Trade_Score_Econ"]
        + 0.40 * df["Trade_Score_ML"]
    )

    # Layer 3
    df = apply_hard_gates(df)
    df["Trade_Score_Final"] = df["Trade_Score_Combined"] * df["Gate_Pass"]

    # Rank within each period
    df["Corridor_Rank"] = (
        df.groupby(["Year", "Month"])["Trade_Score_Final"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Flags
    df["Top_Corridor"]    = (df["Corridor_Rank"] == 1).astype(int)
    df["Recommendation"]  = (df["Corridor_Rank"] <= TM["top_n_corridors"]).astype(int)
    df["Active_Arbitrage"] = (
        (df.get("Arbitrage_Signal", pd.Series(0, index=df.index)) == 1) &
        (df["Gate_Pass"] == 1)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# BUILD OUTPUT TABLE
# ---------------------------------------------------------------------------
def build_output(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """Selects output columns and validates ranking quality."""
    out_cols = [
        "Pair_ID", "Exporter", "Importer", "Year", "Month",
        "MIS", "MIS_MA3",
        "Trade_Score_Econ", "Trade_Score_ML", "Trade_Score_Final",
        "Corridor_Rank", "Gate_Pass", "Gate_Route", "Gate_Friction",
        "Top_Corridor", "Recommendation", "Active_Arbitrage",
        "Route_Feasibility", "Market_Friction_Index",
        "Supply_Confidence", "Arbitrage_Profit_USD_kg",
        "Expected_Margin_USD_kg", "Food_Security_Risk_Flag",
    ]
    avail = [c for c in out_cols if c in df_ranked.columns]
    out = df_ranked[avail].copy()

    # Validate on test set
    test = df_ranked[df_ranked["Year"].isin(TEST_YEARS)]
    if len(test) > 0 and "Exporter_Export_tonnes" in test.columns:
        rank_correlation(
            test["Trade_Score_Final"],
            test["Exporter_Export_tonnes"],
            context="TradeMatching_Test"
        )

    # Summary
    top_freq = out[out["Top_Corridor"] == 1].groupby("Pair_ID").size().sort_values(ascending=False)
    print("\n[03] Months as top-ranked corridor:")
    for corridor, n in top_freq.items():
        print(f"     {corridor}: {n} months")

    path = os.path.join(PRED_DIR, "corridor_rankings.csv")
    out.to_csv(path, index=False)
    print(f"\n     Saved → {path}  ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(data_path: str = None,
        supply_pred_path: str = None,
        forward_mis_path: str = None) -> pd.DataFrame:
    print("=" * 70)
    print("  AATIP | Engine 03: Trade Matching Engine")
    print("=" * 70)

    if data_path is None:
        data_path = os.path.join(BASE_DIR, CFG["data"]["master_csv"])

    df = pd.read_csv(data_path)
    print(f"\n[03] Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # Merge upstream outputs
    if supply_pred_path and os.path.exists(supply_pred_path):
        sup = pd.read_csv(supply_pred_path)
        df = df.merge(
            sup[["Country", "Year", "Month", "Pred_Surplus_Score"]].rename(
                columns={"Country": "Exporter",
                          "Pred_Surplus_Score": "E01_Surplus_Pred"}
            ),
            on=["Exporter", "Year", "Month"], how="left"
        )
        print(f"     Engine 01 merged")

    if forward_mis_path and os.path.exists(forward_mis_path):
        fwd = pd.read_csv(forward_mis_path)
        h6 = fwd[fwd["Forecast_Horizon_M"] == 6][
            ["Pair_ID", "Base_Year", "Base_Month", "Forward_MIS"]
        ].rename(columns={"Base_Year": "Year", "Base_Month": "Month"})
        df = df.merge(h6, on=["Pair_ID", "Year", "Month"], how="left")
        print(f"     Forward MIS merged "
              f"(coverage: {df['Forward_MIS'].notna().mean():.1%})")

    # Fit Lasso
    lasso_pipe, coef_df, medians = fit_lasso(df)

    # Rank corridors
    df_ranked = rank_corridors(df, lasso_pipe, medians)
    out = build_output(df_ranked)

    # Save Lasso artifact
    if lasso_pipe is not None:
        with open(os.path.join(ART_DIR, "trade_matching_lasso.pkl"), "wb") as f:
            pickle.dump({"pipe": lasso_pipe, "medians": medians,
                          "features": safe_features(TRADE_LASSO_FEATURES,
                                                    df.columns.tolist())}, f)

    # Governance card
    card = {
        "engine": "03_trade_matching_engine",
        "layer_hierarchy": [
            "Layer 1: Economic composite (floor) — WB elasticity weights",
            "Layer 2: LassoCV refinement (ranking) — temporal CV, log-transform target",
            "Layer 3: Hard rule gates (filters) — override layers 1 and 2",
        ],
        "final_score_blend": "0.60 × Econ + 0.40 × ML",
        "economic_weights":  {
            "MIS_MA3": TM["w_mis_ma3"],
            "Supply_Confidence": TM["w_supply_confidence"],
            "Route_Feasibility": TM["w_route_feasibility"],
            "Market_Friction_Index": f"-{TM['w_market_friction']}",
        },
        "hard_gates": {
            "Route_Feasibility": f">= {TM['min_route_feasibility']}",
            "Market_Friction_Index": f"<= {TM['max_market_friction']}",
        },
        "top_n": TM["top_n_corridors"],
    }
    with open(os.path.join(GOV_DIR, "03_trade_matching_engine.json"), "w") as f:
        json.dump(card, f, indent=2)

    print("\n[03] Complete ✓")
    print("=" * 70)
    return df_ranked


if __name__ == "__main__":
    run()
