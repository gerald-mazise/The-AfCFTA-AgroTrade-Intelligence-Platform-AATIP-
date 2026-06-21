# =============================================================================
# AATIP — AfCFTA AgroTrade Intelligence Platform
# config/features.py  |  Feature registry — verified against dataset
# =============================================================================
# ALL 144 columns confirmed present before listing here.
# Adding/removing a feature means editing this file — nowhere else.
# validate_registry() catches drift between this file and the actual data.

# ---------------------------------------------------------------------------
# LEAKAGE GUARD — columns that encode future information or policy outputs.
# These must NEVER appear as inputs to any predictive model.
# ---------------------------------------------------------------------------
LEAKAGE_GUARD = [
    "Policy_MIS_20pct_Border_Reduction",
    "Policy_MIS_10pct_Transport_Reduction",
    "Policy_MIS_AfCFTA_Full",
    "Policy_Uplift_S1",
    "Policy_Uplift_S2",
    "Policy_Uplift_AfCFTA",
    "Est_Trade_Increase_S1_pct",
    "Est_Trade_Increase_S2_pct",
    "Est_Trade_Increase_AfCFTA_pct",
    # Econometric outputs computed over full series — not available at prediction time
    "Coint_Flag_36M",
    "Coint_Pval_36M",
    "Half_Life_Price_Reversion_M",
]

# ---------------------------------------------------------------------------
# ID / INDEX COLUMNS — not features
# ---------------------------------------------------------------------------
ID_COLS = ["Pair_ID", "Exporter", "Importer", "Year", "Month"]

# ---------------------------------------------------------------------------
# ENGINE 01 — SUPPLY & DEFICIT CLASSIFIER
# Target: Exporter_Surplus_Score  ∈ {-1 deficit, 0 neutral, 1 surplus}
# Scope: country-month view (dedup from pairwise on Exporter side)
# ---------------------------------------------------------------------------
SUPPLY_FEATURES = [
    # Supply fundamentals
    "Exporter_Supply_Shock_Z",
    "Exporter_Supply_Index",
    "Exporter_Production_YoY_Growth",
    "Exporter_Net_Supply_tonnes",
    "Exporter_Export_Intensity",
    # Climate signals
    "Exporter_Rainfall_Anomaly_Z",
    "Exporter_Drought_Index",
    "Exporter_Climate_Risk_Score",
    "Exporter_Precip_3M",
    "Exporter_Heat_Stress",
    # Harvest seasonality
    "Exporter_Is_Harvest_Month",
    "Exporter_Post_Harvest_3M",
    "Exporter_Pre_Harvest_Scarcity",
    # Price signals (normalised — no absolute prices)
    "Exporter_Price_Zscore",
    "Exporter_Seasonal_Price_Dev",
    "Exporter_Price_Vol_6M",
    # Agricultural fundamentals
    "Exporter_Yield_hg_ha",
    "Exporter_Area_ha",
]
SUPPLY_TARGET = "Exporter_Surplus_Score"

# ---------------------------------------------------------------------------
# ENGINE 02 — MARKET DYNAMICS
# Targets: per-country wholesale prices (exporter and importer separately)
# ECM_Residual is a key feature — pulls forecasts toward long-run equilibrium
# ---------------------------------------------------------------------------
PRICE_FORECAST_FEATURES = [
    # Lagged price gaps (temporal structure)
    "Price_Gap_Lag1",
    "Price_Gap_Lag2",
    "Price_Gap_Lag3",
    # Moving averages
    "Exporter_Price_MA3",
    "Exporter_Price_MA6",
    "Exporter_Price_MA12",
    # Volatility
    "Exporter_Price_Vol_6M",
    # Climate (leading indicator for price)
    "Exporter_Rainfall_Anomaly_Z",
    "Exporter_Drought_Index",
    "Exporter_Heat_Stress",
    # Harvest structure
    "Exporter_Is_Harvest_Month",
    "Exporter_Post_Harvest_3M",
    "Exporter_Pre_Harvest_Scarcity",
    # Supply shocks
    "Exporter_Supply_Shock_Z",
    # Econometric mean-reversion signal
    "ECM_Residual",
    # MIS dynamics
    "MIS_Lag1",
    "Convergence_Speed",
    # Calendar
    "Month",
]

MIS_FEATURES = [
    "MIS", "MIS_MA3", "MIS_MA6", "MIS_Zscore",
    "MIS_Lag1", "MIS_Lag2", "MIS_Persistence_Months",
    "Convergence_Speed", "Convergence_Speed_MA3",
    "ECM_Residual",
    "Price_Corr_12M", "Price_Corr_24M",
    "Price_Transmission_Beta",
]

# ---------------------------------------------------------------------------
# ENGINE 03 — TRADE MATCHING
# Layer 1 uses TRADE_ECONOMIC_FEATURES (theory-grounded)
# Layer 2 uses TRADE_LASSO_FEATURES (broader, ML-selected)
# ---------------------------------------------------------------------------
TRADE_ECONOMIC_FEATURES = [
    "MIS_MA3",           # smoothed MIS — core signal
    "Supply_Confidence",
    "Route_Feasibility",
    "Market_Friction_Index",
    "Total_Logistics_Cost_USD_kg",
    "Transport_Cost_USD_kg",
    "Border_Friction_Cost_USD_kg",
]

TRADE_LASSO_FEATURES = [
    # MIS and convergence
    "MIS", "MIS_MA3", "MIS_MA6", "MIS_Zscore",
    "Price_Gap_USD_kg", "Price_Ratio", "Price_Spread_Pct",
    # Supply/demand balance
    "Supply_Confidence", "Net_Supply_Balance_tonnes",
    "Relative_Production_Advantage", "Importer_Deficit_Severity",
    # Logistics and feasibility
    "Route_Feasibility", "Market_Friction_Index",
    "Total_Logistics_Cost_USD_kg", "Border_Days", "Distance_km",
    "Transport_Cost_USD_kg", "Border_Friction_Cost_USD_kg",
    # Regional agreements
    "Shared_EAC", "Shared_SADC", "Shared_REC",
    "AfCFTA_Priority_Corridor", "Landlocked_Exporter", "Transit_Required",
    # Trade stability
    "Exporter_Trade_Stability_6M", "Importer_Trade_Stability_6M",
    "Exporter_Corridor_Reliability", "Importer_Corridor_Reliability",
    "Formal_Trade_Intensity",
    # Arbitrage and signals
    "Arbitrage_Signal", "Arbitrage_Profit_USD_kg",
    "Exporter_Surplus_Score", "Importer_Surplus_Score",
    "Exporter_Supply_Index", "Importer_Supply_Index",
    "Exporter_Export_Intensity",
    # Calendar
    "Month",
]
TRADE_LASSO_TARGET = "Exporter_Export_tonnes"

# Hard gate features (binary threshold checks — not model inputs)
HARD_GATE_FEATURES = ["Route_Feasibility", "Market_Friction_Index"]

# ---------------------------------------------------------------------------
# ENGINE 04 — INFORMAL TRADE
# ---------------------------------------------------------------------------
INFORMAL_TRIGGER_FEATURES = [
    "MIS",
    "Formal_Trade_Intensity",
    "Price_Convergence_Failure",
]

INFORMAL_PROBABILITY_FEATURES = [
    "MIS_Zscore",
    "MIS_Persistence_Months",
    "Hidden_Trade_Probability",
    "Arbitrage_Profit_USD_kg",
    "Informal_Trade_Signal",
]

# ---------------------------------------------------------------------------
# ENGINE 05 — POLICY SIMULATION
# ---------------------------------------------------------------------------
POLICY_INPUTS = [
    "MIS",
    "Price_Gap_USD_kg",
    "Transport_Cost_USD_kg",
    "Border_Friction_Cost_USD_kg",
    "Total_Logistics_Cost_USD_kg",
    "Market_Friction_Index",
    "Border_Days",
    "Formal_Trade_Intensity",
]

# Pre-computed columns used ONLY for cross-validation (not model inputs)
POLICY_PRECOMPUTED = [
    "Policy_MIS_20pct_Border_Reduction",
    "Policy_MIS_10pct_Transport_Reduction",
    "Policy_MIS_AfCFTA_Full",
    "Est_Trade_Increase_S1_pct",
    "Est_Trade_Increase_S2_pct",
    "Est_Trade_Increase_AfCFTA_pct",
]

# ---------------------------------------------------------------------------
# ENGINE 06 — VALIDATION
# ---------------------------------------------------------------------------
ECONOMETRIC_FEATURES = [
    "Price_Corr_12M", "Price_Corr_24M",
    "ECM_Residual", "Price_Transmission_Beta",
    "Granger_PVal_Exp_to_Imp", "Granger_Causality_Signal",
    "Market_Integration_Score", "AfCFTA_Readiness_Score",
]

# ---------------------------------------------------------------------------
# DASHBOARD DISPLAY SUBSET
# ---------------------------------------------------------------------------
DASHBOARD_CORE = [
    "MIS", "MIS_MA3", "Price_Gap_USD_kg", "Arbitrage_Profit_USD_kg",
    "Trade_Score", "Trade_Opportunity_Score",
    "Supply_Confidence", "Route_Feasibility", "Market_Friction_Index",
    "Food_Security_Risk_Flag", "Crisis_Signal",
    "Formal_Trade_Intensity",
]

FEATURE_LABELS = {
    "MIS": "Market Inefficiency Score",
    "MIS_MA3": "MIS (3-Month Avg)",
    "Price_Gap_USD_kg": "Price Gap (USD/kg)",
    "Arbitrage_Profit_USD_kg": "Arbitrage Profit (USD/kg)",
    "Trade_Score": "Trade Opportunity Score",
    "Supply_Confidence": "Supply Confidence",
    "Route_Feasibility": "Route Feasibility",
    "Market_Friction_Index": "Market Friction Index",
    "Food_Security_Risk_Flag": "Food Security Risk",
    "Crisis_Signal": "Crisis Signal",
    "Formal_Trade_Intensity": "Formal Trade Intensity",
    "Hidden_Trade_Probability": "Hidden Trade Probability",
    "AfCFTA_Readiness_Score": "AfCFTA Readiness Score",
    "Market_Integration_Score": "Market Integration Score",
    "ECM_Residual": "ECM Residual",
    "Price_Transmission_Beta": "Price Transmission β",
}

# ---------------------------------------------------------------------------
# REGISTRY VALIDATION
# ---------------------------------------------------------------------------
def validate_registry(df_columns: list, verbose: bool = True) -> dict:
    """
    Validates all feature lists against actual DataFrame columns.
    Raises ValueError on leakage violations.
    Returns dict of {list_name: [missing_cols]}.
    """
    all_lists = {
        "SUPPLY_FEATURES":            SUPPLY_FEATURES,
        "PRICE_FORECAST_FEATURES":    PRICE_FORECAST_FEATURES,
        "TRADE_ECONOMIC_FEATURES":    TRADE_ECONOMIC_FEATURES,
        "TRADE_LASSO_FEATURES":       TRADE_LASSO_FEATURES,
        "INFORMAL_PROBABILITY_FEATURES": INFORMAL_PROBABILITY_FEATURES,
        "POLICY_INPUTS":              POLICY_INPUTS,
    }

    col_set = set(df_columns)
    missing_report = {}

    for name, feats in all_lists.items():
        missing = [f for f in feats if f not in col_set]
        if missing:
            missing_report[name] = missing
            if verbose:
                print(f"  [WARN] {name}: missing {missing}")
        elif verbose:
            print(f"  [OK]   {name}: {len(feats)} features all present")

    # Leakage check
    all_inputs = [f for feats in all_lists.values() for f in feats]
    leakage_hits = [f for f in all_inputs if f in LEAKAGE_GUARD]
    if leakage_hits:
        raise ValueError(f"LEAKAGE VIOLATION in feature registry: {leakage_hits}")

    if verbose:
        status = "CLEAN" if not missing_report else f"{len(missing_report)} list(s) have gaps"
        print(f"  Registry validation: {status}")

    return missing_report
