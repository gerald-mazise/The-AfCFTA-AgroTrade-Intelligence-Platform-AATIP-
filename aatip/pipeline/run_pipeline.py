# =============================================================================
# AATIP — pipeline/run_pipeline.py  |  Full Pipeline Orchestrator
# =============================================================================
# Runs engines 01–06 in strict sequence.
# Uses importlib to load modules with numeric prefixes (01_supply_engine etc.)
# Degrades gracefully: a failed engine logs the error and pipeline continues.
# Merges all outputs → AATIP_Final_Intelligence.csv
# =============================================================================

import os, sys, json, time, traceback, importlib.util
import pandas as pd
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

PRED_DIR = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
REP_DIR  = os.path.join(BASE_DIR, CFG["data"]["reports_dir"])


def load_engine(filename: str):
    """Loads an engine module by filename using importlib (handles numeric prefixes)."""
    path = os.path.join(BASE_DIR, "models", filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py",""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_engine(name: str, filename: str, kwargs: dict) -> tuple:
    """
    Loads and runs one engine.
    Returns (result, status_dict).
    """
    t0 = time.time()
    try:
        engine = load_engine(filename)
        result = engine.run(**kwargs)
        duration = round(time.time() - t0, 1)
        print(f"\n  ✓ {name} completed in {duration}s")
        return result, {"status": "SUCCESS", "duration_s": duration}
    except Exception as e:
        duration = round(time.time() - t0, 1)
        tb = traceback.format_exc()
        print(f"\n  ✗ {name} FAILED: {e}")
        print(f"    {tb.splitlines()[-1]}")
        return None, {"status": "FAILED", "error": str(e),
                       "traceback": tb, "duration_s": duration}


def merge_all_outputs(data_path: str) -> pd.DataFrame:
    """Merges master dataset with all prediction outputs."""
    print("\n>>> MERGING ALL OUTPUTS")
    df = pd.read_csv(data_path)
    print(f"    Base: {df.shape}")

    merges = [
        ("corridor_rankings.csv",        ["Pair_ID","Year","Month"]),
        ("informal_trade_estimates.csv", ["Pair_ID","Year","Month"]),
        ("policy_scenarios.csv",         ["Pair_ID","Year","Month"]),
    ]
    for fname, keys in merges:
        path = os.path.join(PRED_DIR, fname)
        if not os.path.exists(path):
            print(f"    [SKIP] {fname} — not found")
            continue
        pred = pd.read_csv(path)
        # Drop columns already in df (except merge keys)
        dup = [c for c in pred.columns if c in df.columns and c not in keys]
        pred = pred.drop(columns=dup, errors="ignore")
        df = df.merge(pred, on=keys, how="left")
        print(f"    Merged {fname} → {df.shape}")

    # Price forecasts (exporter, horizon=6)
    fc_path = os.path.join(PRED_DIR, "price_forecasts.csv")
    if os.path.exists(fc_path):
        fc = pd.read_csv(fc_path)
        fc6 = fc[fc["Horizon_M"] == 6][[
            "Country","Forecast_Year","Forecast_Month",
            "Price_Forecast","CI_Lower","CI_Upper"
        ]].rename(columns={
            "Country": "Exporter",
            "Forecast_Year": "Year",
            "Forecast_Month": "Month",
            "Price_Forecast": "Exp_Price_FC_6M",
            "CI_Lower": "Exp_Price_FC_Low",
            "CI_Upper": "Exp_Price_FC_High",
        })
        df = df.merge(fc6, on=["Exporter","Year","Month"], how="left")
        print(f"    Merged price_forecasts.csv (h=6) → {df.shape}")

    out_path = os.path.join(BASE_DIR, "AATIP_Final_Intelligence.csv")
    df.to_csv(out_path, index=False)
    print(f"\n    AATIP_Final_Intelligence.csv → {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def run_full_pipeline(data_path: str = None) -> dict:
    if data_path is None:
        data_path = os.path.join(BASE_DIR, CFG["data"]["master_csv"])

    print("\n" + "═" * 70)
    print("  AATIP — FULL PIPELINE EXECUTION")
    print("  AfCFTA AgroTrade Intelligence Platform")
    print("═" * 70)
    print(f"  Data: {os.path.basename(data_path)}")

    log = {}
    sup_path  = os.path.join(PRED_DIR, "surplus_deficit_predictions.csv")
    fwd_path  = os.path.join(PRED_DIR, "forward_mis.csv")
    rank_path = os.path.join(PRED_DIR, "corridor_rankings.csv")

    # Engine 01
    print("\n\n>>> ENGINE 01 — Supply & Deficit Classifier")
    _, log["01"] = run_engine(
        "01_supply_engine", "01_supply_engine.py",
        {"data_path": data_path}
    )

    # Engine 02
    print("\n\n>>> ENGINE 02 — Market Dynamics Engine")
    _, log["02"] = run_engine(
        "02_market_dynamics_engine", "02_market_dynamics_engine.py",
        {"data_path": data_path,
         "supply_predictions_path": sup_path if os.path.exists(sup_path) else None}
    )

    # Engine 03
    print("\n\n>>> ENGINE 03 — Trade Matching Engine")
    _, log["03"] = run_engine(
        "03_trade_matching_engine", "03_trade_matching_engine.py",
        {"data_path": data_path,
         "supply_pred_path": sup_path if os.path.exists(sup_path) else None,
         "forward_mis_path": fwd_path if os.path.exists(fwd_path) else None}
    )

    # Engine 04
    print("\n\n>>> ENGINE 04 — Informal Trade Engine")
    _, log["04"] = run_engine(
        "04_informal_trade_engine", "04_informal_trade_engine.py",
        {"data_path": data_path,
         "rankings_path": rank_path if os.path.exists(rank_path) else None}
    )

    # Engine 05
    print("\n\n>>> ENGINE 05 — Policy Simulation Engine")
    _, log["05"] = run_engine(
        "05_policy_simulation_engine", "05_policy_simulation_engine.py",
        {"data_path": data_path}
    )

    # Engine 06
    print("\n\n>>> ENGINE 06 — Validation & Econometrics")
    _, log["06"] = run_engine(
        "06_validation_econometrics", "06_validation_econometrics.py",
        {"data_path": data_path}
    )

    # Merge
    try:
        merge_all_outputs(data_path)
        log["merge"] = {"status": "SUCCESS"}
    except Exception as e:
        log["merge"] = {"status": "FAILED", "error": str(e)}
        print(f"  ✗ Merge failed: {e}")

    # Summary
    print("\n" + "═" * 70)
    print("  PIPELINE SUMMARY")
    print("═" * 70)
    for key, result in log.items():
        s = result.get("status","?")
        d = f"  ({result.get('duration_s','')}s)" if "duration_s" in result else ""
        print(f"  {key:<10} {s}{d}")

    n_ok = sum(1 for r in log.values() if r.get("status") == "SUCCESS")
    print(f"\n  {n_ok}/{len(log)} steps successful")
    print("═" * 70)

    log_path = os.path.join(REP_DIR, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  Pipeline log → {log_path}")

    return log


if __name__ == "__main__":
    run_full_pipeline()
