# =============================================================================
# AATIP — models/05_policy_simulation_engine.py  |  Policy Simulation Engine
# =============================================================================
# Quantifies trade impact of AfCFTA policy interventions.
# No ML fitting — this is a structural economic simulation engine.
#
# Three canonical scenarios (from config):
#   Baseline:      No intervention
#   AfCFTA_Phase1: -20% border friction (single-window customs, AU 2027 target)
#   AfCFTA_Full:   -50% border, -10% transport (AU 2035 full implementation)
#
# Sensitivity: sweeps elasticity 1.0–3.5 to bound uncertainty.
# Validation: checks simulated MIS against pre-computed policy columns.
#
# OUTPUT: outputs/predictions/policy_scenarios.csv
#         outputs/reports/policy_headline_numbers.json
#         outputs/reports/policy_sensitivity.csv
# =============================================================================

import os, sys, json, warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

PRED_DIR = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
REP_DIR  = os.path.join(BASE_DIR, CFG["data"]["reports_dir"])
GOV_DIR  = os.path.join(BASE_DIR, "governance", "model_cards")
for d in [PRED_DIR, REP_DIR, GOV_DIR]:
    os.makedirs(d, exist_ok=True)

PS  = CFG["policy_simulation"]
MIS = CFG["mis"]


# ---------------------------------------------------------------------------
# CORE SIMULATION
# ---------------------------------------------------------------------------
def simulate_scenario(df: pd.DataFrame,
                       border_red: float,
                       transport_red: float,
                       elasticity: float,
                       name: str) -> pd.DataFrame:
    """
    Simulates one AfCFTA policy scenario.

    Mechanics:
      1. Reduce logistics costs by scenario parameters
      2. Recompute MIS using new logistics costs
      3. Estimate ΔTrade% = elasticity × |ΔLogistics%|
      4. Estimate incremental volume and USD value

    Args:
        border_red:     fraction by which border friction cost falls (e.g. 0.20)
        transport_red:  fraction by which transport cost falls (e.g. 0.10)
        elasticity:     % trade increase per 1% logistics cost reduction
        name:           scenario name (used as column suffix)

    Returns DataFrame with scenario columns added.
    """
    df = df.copy()

    orig_t = df["Transport_Cost_USD_kg"].fillna(0.062)
    orig_b = df["Border_Friction_Cost_USD_kg"].fillna(0.045)
    orig_L = orig_t + orig_b

    new_t = orig_t * (1 - transport_red)
    new_b = orig_b * (1 - border_red)
    new_L = new_t + new_b

    # Recomputed MIS
    gap = df["Price_Gap_USD_kg"].fillna(0)
    new_mis = np.where(new_L > 0, (gap - new_L) / new_L, df["MIS"].fillna(0))

    df[f"New_MIS_{name}"]      = new_mis.round(4)
    df[f"Delta_MIS_{name}"]    = (new_mis - df["MIS"].fillna(0)).round(4)
    df[f"New_Logistics_{name}"] = new_L.round(5)

    # Trade volume change
    delta_L_pct = np.where(
        orig_L > 0, (new_L - orig_L) / orig_L, 0
    )
    delta_trade_pct = elasticity * np.abs(delta_L_pct)
    df[f"Trade_Increase_Pct_{name}"] = (delta_trade_pct * 100).round(3)

    # Volume estimate
    formal_vol = df.get(
        "Exporter_Export_tonnes",
        pd.Series(5000.0, index=df.index)
    ).fillna(5000.0).clip(lower=0)

    df[f"Inc_Vol_tonnes_{name}"] = (formal_vol * delta_trade_pct).round(0)

    # USD value: incremental tonnes × price × 1000 (kg→tonne conversion)
    price_kg = df["Exporter_Price_Wholesale_USD_kg"].fillna(0.35)
    df[f"Inc_Value_USD_{name}"] = (
        df[f"Inc_Vol_tonnes_{name}"] * price_kg * 1000
    ).round(0)

    # Month unlocked: MIS was ≤ 0 and becomes > 0
    was_inactive = df["MIS"].fillna(0) <= MIS["arbitrage_threshold"]
    now_active   = pd.Series(new_mis, index=df.index) > MIS["arbitrage_threshold"]
    df[f"Month_Unlocked_{name}"] = (was_inactive & now_active).astype(int)

    return df


# ---------------------------------------------------------------------------
# CROSS-VALIDATE AGAINST PRE-COMPUTED COLUMNS
# ---------------------------------------------------------------------------
def cross_validate(df: pd.DataFrame, scenario_name: str) -> dict:
    """
    Validates simulation output against pre-computed Policy_MIS_* columns.
    These provide a data-grounded sanity check on the simulation mechanics.
    """
    precomputed_map = {
        "AfCFTA_Phase1": "Policy_MIS_20pct_Border_Reduction",
        "AfCFTA_Full":   "Policy_MIS_AfCFTA_Full",
    }
    col_pre = precomputed_map.get(scenario_name)
    col_sim = f"New_MIS_{scenario_name}"

    if col_pre not in df.columns or col_sim not in df.columns:
        return {}

    mask = df[[col_sim, col_pre]].notna().all(axis=1)
    n = mask.sum()
    if n < 5:
        return {"n": int(n), "note": "insufficient_overlap"}

    sim = df.loc[mask, col_sim]
    pre = df.loc[mask, col_pre]

    corr = float(sim.corr(pre))
    mae  = float(np.abs(sim - pre).mean())

    flag = "PASS" if corr >= 0.85 else "WARN"
    print(f"     [{flag}] {scenario_name} cross-val: "
          f"corr={corr:.3f}, MAE={mae:.5f}, n={n}")
    return {"corr": round(corr, 4), "mae": round(mae, 6), "n": int(n)}


# ---------------------------------------------------------------------------
# SENSITIVITY SWEEP
# ---------------------------------------------------------------------------
def sensitivity_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Sweeps elasticity across the configured range for AfCFTA_Full scenario."""
    print("\n[05] Sensitivity sweep (AfCFTA_Full, elasticity range)...")

    scenario = PS["scenarios"]["AfCFTA_Full"]
    elasticities = np.linspace(
        PS["elasticity_min"], PS["elasticity_max"], PS["elasticity_steps"]
    )
    records = []

    for e in elasticities:
        sim = simulate_scenario(
            df,
            border_red=scenario["border_reduction"],
            transport_red=scenario["transport_reduction"],
            elasticity=e,
            name="sens",
        )
        records.append({
            "Elasticity":                     round(float(e), 2),
            "Total_Inc_Vol_tonnes":           int(sim["Inc_Vol_tonnes_sens"].sum()),
            "Total_Inc_Value_USD":            int(sim["Inc_Value_USD_sens"].sum()),
            "Months_Unlocked":                int(sim["Month_Unlocked_sens"].sum()),
        })

    sens_df = pd.DataFrame(records)
    vol_range = (sens_df["Total_Inc_Vol_tonnes"].min(),
                 sens_df["Total_Inc_Vol_tonnes"].max())
    print(f"     Trade increase range: {vol_range[0]:,} – {vol_range[1]:,} tonnes "
          f"(elasticity {PS['elasticity_min']}–{PS['elasticity_max']})")
    return sens_df


# ---------------------------------------------------------------------------
# HEADLINE NUMBERS
# ---------------------------------------------------------------------------
def build_headlines(results: dict) -> dict:
    """Corridor-level and overall summaries for dashboard."""
    headlines = {}

    for name, df_sim in results.items():
        vol_col  = f"Inc_Vol_tonnes_{name}"
        val_col  = f"Inc_Value_USD_{name}"
        unlk_col = f"Month_Unlocked_{name}"

        if vol_col not in df_sim.columns:
            continue

        corridor_summary = {}
        for corridor, grp in df_sim.groupby("Pair_ID"):
            corridor_summary[corridor] = {
                "inc_vol_tonnes": int(grp[vol_col].sum()),
                "inc_value_usd":  int(grp[val_col].sum()),
                "months_unlocked": int(grp[unlk_col].sum()),
                "mean_mis_delta": round(float(grp[f"Delta_MIS_{name}"].mean()), 4),
            }

        headlines[name] = {
            "total_inc_vol_tonnes":     int(df_sim[vol_col].sum()),
            "total_inc_value_usd":      int(df_sim[val_col].sum()),
            "total_months_unlocked":    int(df_sim[unlk_col].sum()),
            "corridors_activated":      int(
                df_sim.groupby("Pair_ID")[unlk_col].sum().gt(0).sum()
            ),
            "by_corridor": corridor_summary,
        }

    return headlines


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(data_path: str = None) -> pd.DataFrame:
    print("=" * 70)
    print("  AATIP | Engine 05: Policy Simulation Engine")
    print("=" * 70)

    if data_path is None:
        data_path = os.path.join(BASE_DIR, CFG["data"]["master_csv"])

    df = pd.read_csv(data_path)
    print(f"\n[05] Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # Run all three scenarios at base elasticity
    results = {}
    validations = {}

    for name, scenario in PS["scenarios"].items():
        print(f"\n[05] Scenario: {name}")
        print(f"     {scenario['description']}")
        print(f"     Border Δ: -{scenario['border_reduction']*100:.0f}%  "
              f"Transport Δ: -{scenario['transport_reduction']*100:.0f}%  "
              f"Elasticity: {PS['elasticity_base']}")

        df = simulate_scenario(
            df,
            border_red=scenario["border_reduction"],
            transport_red=scenario["transport_reduction"],
            elasticity=PS["elasticity_base"],
            name=name,
        )
        validations[name] = cross_validate(df, name)
        results[name] = df.copy()

    # Save combined output (all scenario columns on same DataFrame)
    base_cols = [
        "Pair_ID","Exporter","Importer","Year","Month",
        "MIS","Price_Gap_USD_kg","Transport_Cost_USD_kg",
        "Border_Friction_Cost_USD_kg",
    ]
    scenario_cols = []
    for name in PS["scenarios"]:
        scenario_cols += [
            f"New_MIS_{name}", f"Delta_MIS_{name}",
            f"Trade_Increase_Pct_{name}",
            f"Inc_Vol_tonnes_{name}", f"Inc_Value_USD_{name}",
            f"Month_Unlocked_{name}",
        ]
    all_cols = [c for c in base_cols + scenario_cols if c in df.columns]
    out = df[all_cols].copy()

    path = os.path.join(PRED_DIR, "policy_scenarios.csv")
    out.to_csv(path, index=False)
    print(f"\n     Saved → {path}  ({len(out)} rows)")

    # Headline numbers
    headlines = build_headlines(results)

    hl_path = os.path.join(REP_DIR, "policy_headline_numbers.json")
    with open(hl_path, "w") as f:
        json.dump(headlines, f, indent=2, default=str)
    print(f"     Headlines saved → {hl_path}")

    # Print headline table
    print("\n" + "=" * 60)
    print(f"  POLICY HEADLINE RESULTS  (elasticity={PS['elasticity_base']})")
    print("=" * 60)
    for name, nums in headlines.items():
        print(f"\n  {name}")
        print(f"    Incremental volume: {nums['total_inc_vol_tonnes']:>12,} tonnes")
        print(f"    Incremental value:  ${nums['total_inc_value_usd']:>12,}")
        print(f"    Months unlocked:    {nums['total_months_unlocked']:>12}")

    # Sensitivity sweep
    sens_df = sensitivity_sweep(df)
    sens_path = os.path.join(REP_DIR, "policy_sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)
    print(f"\n     Sensitivity table saved → {sens_path}")

    # Governance card
    card = {
        "engine": "05_policy_simulation_engine",
        "scenarios": {k: v["description"] for k, v in PS["scenarios"].items()},
        "base_elasticity": PS["elasticity_base"],
        "elasticity_range": [PS["elasticity_min"], PS["elasticity_max"]],
        "cross_validation": validations,
        "caveats": [
            "Elasticity assumed constant — real response varies by corridor, season, commodity",
            "Volume estimates assume current price levels (large-scale reforms change prices)",
            "AfCFTA_Full assumes infrastructure co-investment — not guaranteed",
            "Simulation is partial equilibrium — general equilibrium effects not modelled",
        ],
    }
    with open(os.path.join(GOV_DIR, "05_policy_simulation_engine.json"), "w") as f:
        json.dump(card, f, indent=2)

    print("\n[05] Complete ✓")
    print("=" * 70)
    return out


if __name__ == "__main__":
    run()
