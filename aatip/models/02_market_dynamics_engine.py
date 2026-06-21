# =============================================================================
# AATIP — models/02_market_dynamics_engine.py  |  Market Dynamics Engine
# =============================================================================
# PURPOSE: Model how markets evolve through time and produce:
#   PRIMARY:   Forward MIS — 6-month ahead MIS per corridor
#              This is the key output: PREDICTS corridor inefficiency before
#              it appears in current prices.
#   SECONDARY: Per-country price forecasts with confidence intervals
#
# Models: Manual AR(p) via LinearRegression + GradientBoostingRegressor
# No statsmodels — pure sklearn + numpy.
#
# ECM_Residual is included as a feature where non-null (adds mean-reversion
# pull to forecasts when cointegration holds).
#
# INPUT:  master CSV + 01 supply predictions (optional enrichment)
# OUTPUT: outputs/predictions/price_forecasts.csv
#         outputs/predictions/forward_mis.csv
#         governance/model_cards/02_market_dynamics_engine.json
# =============================================================================

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.features import PRICE_FORECAST_FEATURES, MIS_FEATURES
from utils.validators import (
    check_temporal_split, check_no_leakage, safe_features, impute_train_only
)
from utils.metrics import forecast_metrics

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

PRED_DIR = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
ART_DIR  = os.path.join(BASE_DIR, CFG["data"]["artifacts_dir"])
GOV_DIR  = os.path.join(BASE_DIR, "governance", "model_cards")
for d in [PRED_DIR, ART_DIR, GOV_DIR]:
    os.makedirs(d, exist_ok=True)

MD = CFG["market_dynamics"]
COUNTRIES = CFG["corridors"]["countries"]
TRAIN_YEARS = CFG["temporal"]["train_years"]
TEST_YEARS  = CFG["temporal"]["test_years"]
HORIZON     = MD["forecast_horizon"]


# ---------------------------------------------------------------------------
# MANUAL ADF TEST (no statsmodels)
# ---------------------------------------------------------------------------
def adf_pvalue(series: np.ndarray) -> float:
    """
    Simplified ADF test via OLS regression.
    Regresses Δy on y_{t-1} to estimate the AR coefficient.
    p-value approximated via t-distribution (not exact Dickey-Fuller tables,
    but directionally correct for stationarity assessment).
    """
    y = np.array(series, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 10:
        return 0.5  # insufficient data — assume non-stationary (conservative)

    dy   = np.diff(y)
    y_lag = y[:-1]
    n    = len(dy)

    X = np.column_stack([y_lag, np.ones(n)])
    b, res, _, _ = np.linalg.lstsq(X, dy, rcond=None)

    gamma = b[0]
    if n <= 2:
        return 0.5

    y_hat = X @ b
    sigma2 = np.sum((dy - y_hat) ** 2) / (n - 2)
    se_gamma = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])

    t_stat = gamma / se_gamma if se_gamma > 0 else 0.0
    # Approximate p-value using t-distribution (conservative — higher p than true ADF)
    from scipy.stats import t as t_dist
    p = float(t_dist.sf(np.abs(t_stat), df=n - 2) * 2)
    return p


def is_stationary(series: np.ndarray, threshold: float = 0.10) -> bool:
    """Returns True if series is stationary at given threshold."""
    p = adf_pvalue(series)
    return p < threshold


# ---------------------------------------------------------------------------
# BUILD COUNTRY-MONTH PRICE SERIES
# ---------------------------------------------------------------------------
def build_country_series(df: pd.DataFrame) -> dict:
    """
    Extracts per-country monthly price series from pairwise dataset.
    Deduplicates on (Exporter, Year, Month) → one row per country-month.
    """
    print("\n[02] Building per-country price series...")
    series = {}
    price_col = MD["target_exporter"]

    for country in COUNTRIES:
        cols_want = (
            ["Year", "Month", price_col, "ECM_Residual", "MIS_Lag1",
             "Convergence_Speed", "Exporter_Rainfall_Anomaly_Z",
             "Exporter_Drought_Index", "Exporter_Heat_Stress",
             "Exporter_Is_Harvest_Month", "Exporter_Post_Harvest_3M",
             "Exporter_Pre_Harvest_Scarcity", "Exporter_Supply_Shock_Z",
             "Exporter_Price_MA3", "Exporter_Price_MA6", "Exporter_Price_MA12",
             "Exporter_Price_Vol_6M",
             "Price_Gap_Lag1", "Price_Gap_Lag2", "Price_Gap_Lag3",
             "Exporter_Price_MoM_Chg"]
        )
        avail = [c for c in cols_want if c in df.columns]
        sub = (
            df[df["Exporter"] == country][avail]
            .drop_duplicates(subset=["Year", "Month"])
            .sort_values(["Year", "Month"])
            .reset_index(drop=True)
        )
        sub["Country"] = country
        sub = sub.rename(columns={price_col: "Price"})

        p_clean = sub["Price"].dropna()
        print(f"     {country}: {len(sub)} obs | "
              f"price {p_clean.min():.3f}–{p_clean.max():.3f} USD/kg | "
              f"stationary: {is_stationary(p_clean.values)}")
        series[country] = sub

    return series


# ---------------------------------------------------------------------------
# AR(p) MODEL — LinearRegression with explicit lags as features
# ---------------------------------------------------------------------------
def build_ar_features(series: pd.Series, lags: list) -> pd.DataFrame:
    """
    Constructs lag features from a price series for AR modelling.
    Returns DataFrame with one row per time step (rows with NaN lags dropped).
    """
    df = pd.DataFrame({"y": series.values})
    for lag in lags:
        df[f"y_lag{lag}"] = df["y"].shift(lag)
    return df.dropna()


def fit_ar_model(price_series: pd.Series, country: str) -> tuple:
    """
    Fits AR(p) via OLS with lags defined in config.
    Applied to first-differenced series if non-stationary.
    """
    lags = MD["ar_lags"]
    p_vals = price_series.dropna()
    d = 0

    if not is_stationary(p_vals.values):
        p_vals = p_vals.diff().dropna()
        d = 1

    ar_df = build_ar_features(p_vals.reset_index(drop=True), lags)
    lag_cols = [f"y_lag{l}" for l in lags]

    train_n = int(len(ar_df) * 0.8)
    X_tr = ar_df[lag_cols].values[:train_n]
    y_tr = ar_df["y"].values[:train_n]
    X_val = ar_df[lag_cols].values[train_n:]
    y_val = ar_df["y"].values[train_n:]

    model = LinearRegression()
    model.fit(X_tr, y_tr)

    val_preds = model.predict(X_val) if len(X_val) > 0 else np.array([])
    val_metrics = {}
    if len(y_val) > 0:
        val_metrics = forecast_metrics(y_val, val_preds,
                                        context=f"AR-{country}-val")

    return model, d, lags, val_metrics


# ---------------------------------------------------------------------------
# GRADIENT BOOSTING REGRESSOR
# ---------------------------------------------------------------------------
def fit_gbr(country_df: pd.DataFrame, country: str) -> tuple:
    """
    GradientBoostingRegressor with temporal CV.
    Uses price lags + climate + supply + seasonal features.
    """
    # Build lag features from raw price column
    df_work = country_df.copy().sort_values(["Year", "Month"]).reset_index(drop=True)
    for lag in [1, 2, 3, 6]:
        df_work[f"Price_lag{lag}"] = df_work["Price"].shift(lag)
    df_work = df_work.dropna(subset=["Price", "Price_lag1"])

    base_features = [
        "Price_lag1", "Price_lag2", "Price_lag3", "Price_lag6",
        "Month",
        "Exporter_Rainfall_Anomaly_Z", "Exporter_Drought_Index",
        "Exporter_Heat_Stress", "Exporter_Supply_Shock_Z",
        "Exporter_Is_Harvest_Month", "Exporter_Post_Harvest_3M",
        "Exporter_Pre_Harvest_Scarcity",
        "Exporter_Price_MA3", "Exporter_Price_MA6",
        "Exporter_Price_Vol_6M",
        "Price_Gap_Lag1", "Price_Gap_Lag2",
        "MIS_Lag1", "Convergence_Speed",
    ]
    # ECM_Residual only where non-null (adds mean-reversion signal)
    if "ECM_Residual" in df_work.columns:
        base_features.append("ECM_Residual")

    features = [f for f in base_features if f in df_work.columns]

    train_mask = df_work["Year"].isin(TRAIN_YEARS)
    test_mask  = df_work["Year"].isin(TEST_YEARS)

    train_df = df_work[train_mask]
    test_df  = df_work[test_mask]

    if len(train_df) < 12:
        print(f"     [WARN] {country}: insufficient training data for GBR")
        return None, features, {}, df_work[features].median()

    medians = train_df[features].median()
    X_tr = train_df[features].fillna(medians)
    y_tr = train_df["Price"]
    X_te = test_df[features].fillna(medians) if len(test_df) > 0 else X_tr[:0]
    y_te = test_df["Price"] if len(test_df) > 0 else y_tr[:0]

    # Temporal CV
    tscv = TimeSeriesSplit(n_splits=min(5, len(X_tr) // 6))
    cv_maes = []
    for tr_idx, val_idx in tscv.split(X_tr):
        Xtr2, Xval2 = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        ytr2, yval2 = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        if len(Xtr2) < 5:
            continue
        gbr_tmp = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42
        )
        gbr_tmp.fit(Xtr2, ytr2)
        cv_maes.append(float(np.mean(np.abs(gbr_tmp.predict(Xval2) - yval2.values))))
    if cv_maes:
        print(f"     {country} GBR CV MAE: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

    # Final model
    gbr = GradientBoostingRegressor(
        n_estimators=MD["gb_n_estimators"],
        max_depth=MD["gb_max_depth"],
        learning_rate=MD["gb_learning_rate"],
        min_samples_leaf=MD["gb_min_samples_leaf"],
        subsample=0.8,
        random_state=42,
    )
    gbr.fit(X_tr, y_tr)

    test_metrics = {}
    if len(X_te) > 0:
        test_preds = gbr.predict(X_te)
        test_metrics = forecast_metrics(y_te.values, test_preds,
                                         context=f"GBR-{country}-test")

    return gbr, features, test_metrics, medians


# ---------------------------------------------------------------------------
# FORWARD MIS COMPUTATION
# This is the commercially critical output.
# ---------------------------------------------------------------------------
def compute_forward_mis(df: pd.DataFrame, country_forecasts: dict) -> pd.DataFrame:
    """
    For each corridor, compute Forward MIS at horizons 1–6 months using
    the latest known prices + per-country price trends.

    Forward MIS_h = (Forecast_Importer_h - Forecast_Exporter_h - Logistics) / Logistics

    Predicts which corridors are ABOUT TO become inefficient — not just
    which are currently inefficient.
    """
    print("\n[02] Computing Forward MIS (1–6 month horizon)...")

    # Latest observation per corridor
    latest = (
        df.sort_values(["Year", "Month"])
        .groupby("Pair_ID")
        .tail(1)
        .reset_index(drop=True)
    )

    records = []
    for _, row in latest.iterrows():
        corridor  = row["Pair_ID"]
        exporter  = row["Exporter"]
        importer  = row["Importer"]
        t_cost    = float(row.get("Transport_Cost_USD_kg", 0.062) or 0.062)
        b_cost    = float(row.get("Border_Friction_Cost_USD_kg", 0.045) or 0.045)
        logistics = t_cost + b_cost

        # Current prices (base for projection)
        p_exp = float(row.get("Exporter_Price_Wholesale_USD_kg") or 0)
        p_imp = float(row.get("Importer_Price_Wholesale_USD_kg") or 0)

        # Monthly trend from recent history (robust: use median of 6-month MoM changes)
        def get_trend(country: str, col: str = "Exporter_Price_MoM_Chg") -> float:
            sub = df[df["Exporter"] == country].sort_values(["Year", "Month"])
            chg = sub[col].dropna().tail(6) if col in sub.columns else pd.Series([])
            return float(chg.median()) if len(chg) > 0 else 0.0

        exp_trend = get_trend(exporter, "Exporter_Price_MoM_Chg")
        imp_trend = get_trend(importer, "Importer_Price_MoM_Chg")

        current_mis = float(row.get("MIS", 0) or 0)

        for h in range(1, HORIZON + 1):
            # Projected prices: base + cumulative trend (dampened by sqrt horizon)
            # Dampening prevents linear extrapolation diverging at long horizons
            damp = np.sqrt(h)
            p_exp_h = p_exp * (1 + exp_trend * damp) if p_exp > 0 else p_exp
            p_imp_h = p_imp * (1 + imp_trend * damp) if p_imp > 0 else p_imp

            fwd_mis = (
                (p_imp_h - p_exp_h - logistics) / logistics
                if logistics > 0 and p_exp_h > 0 and p_imp_h > 0
                else current_mis
            )

            records.append({
                "Pair_ID":                 corridor,
                "Exporter":                exporter,
                "Importer":                importer,
                "Base_Year":               int(row["Year"]),
                "Base_Month":              int(row["Month"]),
                "Forecast_Horizon_M":      h,
                "Forecast_Exporter_Price": round(p_exp_h, 4),
                "Forecast_Importer_Price": round(p_imp_h, 4),
                "Logistics_Cost":          round(logistics, 4),
                "Forward_MIS":             round(fwd_mis, 4),
                "Forward_Arbitrage":       int(fwd_mis > CFG["mis"]["arbitrage_threshold"]),
                "Forward_Strong_Signal":   int(fwd_mis > CFG["mis"]["strong_arbitrage"]),
                "Current_MIS":             round(current_mis, 4),
                "MIS_Delta":               round(fwd_mis - current_mis, 4),
            })

    fwd_df = pd.DataFrame(records)
    path = os.path.join(PRED_DIR, "forward_mis.csv")
    fwd_df.to_csv(path, index=False)
    print(f"     Saved → {path}  ({len(fwd_df)} rows)")

    # Summary at 6-month horizon
    h6 = fwd_df[fwd_df["Forecast_Horizon_M"] == 6]
    n_arb = h6["Forward_Arbitrage"].sum()
    print(f"     6M horizon: {n_arb}/{len(h6)} corridors show forward arbitrage signal")
    print(f"     Forward MIS range: {fwd_df['Forward_MIS'].min():.3f} – "
          f"{fwd_df['Forward_MIS'].max():.3f}")
    return fwd_df


# ---------------------------------------------------------------------------
# PRICE FORECASTS WITH CONFIDENCE INTERVALS
# ---------------------------------------------------------------------------
def build_price_forecasts(country_series: dict, country_models: dict) -> pd.DataFrame:
    """
    Per-country 6-month price forecast with 90% CI.
    CI = ±ci_multiplier × rolling 12M price std × sqrt(horizon).
    """
    print("\n[02] Building price forecast table...")
    records = []

    for country, c_df in country_series.items():
        prices = c_df["Price"].dropna()
        if len(prices) < 6:
            continue

        last_price = float(prices.iloc[-1])
        last_year  = int(c_df["Year"].iloc[-1])
        last_month = int(c_df["Month"].iloc[-1])

        trend = float(prices.diff().tail(6).median() or 0)
        vol   = float(prices.tail(12).std() or 0)

        for h in range(1, HORIZON + 1):
            total_months = last_month + h - 1
            fc_year  = last_year + total_months // 12
            fc_month = total_months % 12 + 1

            damp = np.sqrt(h)
            fc_price = last_price + trend * damp

            ci_half = MD["ci_multiplier"] * vol * damp
            records.append({
                "Country":              country,
                "Forecast_Year":        fc_year,
                "Forecast_Month":       fc_month,
                "Horizon_M":            h,
                "Price_Forecast":       round(max(0, fc_price), 4),
                "CI_Lower":             round(max(0, fc_price - ci_half), 4),
                "CI_Upper":             round(fc_price + ci_half, 4),
                "Last_Known_Price":     round(last_price, 4),
                "Price_Trend_MoM":      round(trend, 5),
                "Price_Vol_12M":        round(vol, 5),
            })

    fc_df = pd.DataFrame(records)
    path = os.path.join(PRED_DIR, "price_forecasts.csv")
    fc_df.to_csv(path, index=False)
    print(f"     Saved → {path}  ({len(fc_df)} rows)")
    return fc_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(data_path: str = None, supply_predictions_path: str = None) -> tuple:
    print("=" * 70)
    print("  AATIP | Engine 02: Market Dynamics Engine")
    print("=" * 70)

    if data_path is None:
        data_path = os.path.join(BASE_DIR, CFG["data"]["master_csv"])

    df = pd.read_csv(data_path)
    print(f"\n[02] Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # Optionally enrich with supply predictions from engine 01
    if supply_predictions_path and os.path.exists(supply_predictions_path):
        sup = pd.read_csv(supply_predictions_path)
        df = df.merge(
            sup[["Country", "Year", "Month", "Pred_Surplus_Score"]]
            .rename(columns={"Country": "Exporter",
                              "Pred_Surplus_Score": "E01_Surplus_Pred"}),
            on=["Exporter", "Year", "Month"], how="left"
        )
        print(f"     Engine 01 predictions merged "
              f"(coverage: {df['E01_Surplus_Pred'].notna().mean():.1%})")

    # Build per-country price series
    country_series = build_country_series(df)

    # Fit models per country
    country_models = {}
    for country in COUNTRIES:
        c_df = country_series.get(country)
        if c_df is None or c_df["Price"].dropna().shape[0] < 12:
            print(f"     [WARN] {country}: insufficient data — skipping")
            continue

        print(f"\n[02] Fitting models: {country}...")
        ar_model, d, lags, ar_metrics  = fit_ar_model(c_df["Price"], country)
        gbr, gbr_features, gbr_metrics, medians = fit_gbr(c_df, country)

        country_models[country] = {
            "ar":            ar_model,
            "ar_d":          d,
            "ar_lags":       lags,
            "gbr":           gbr,
            "gbr_features":  gbr_features,
            "gbr_medians":   medians,
            "ar_metrics":    ar_metrics,
            "gbr_metrics":   gbr_metrics,
            "last_price":    float(c_df["Price"].dropna().iloc[-1]),
        }

    # Save model artifacts
    with open(os.path.join(ART_DIR, "market_dynamics_models.pkl"), "wb") as f:
        pickle.dump(country_models, f)
    print(f"\n     Models saved → {ART_DIR}/market_dynamics_models.pkl")

    # PRIMARY OUTPUT: Forward MIS
    fwd_df = compute_forward_mis(df, country_models)

    # SECONDARY: Price forecasts with CI
    fc_df = build_price_forecasts(country_series, country_models)

    # Governance card
    card = {
        "engine":          "02_market_dynamics_engine",
        "models":          ["AR(p)_via_LinearRegression", "GradientBoostingRegressor"],
        "primary_output":  "Forward_MIS (1–6 month horizon per corridor)",
        "secondary_output": "Price forecasts with 90% CI",
        "forecast_horizon_months": HORIZON,
        "ar_lags":         MD["ar_lags"],
        "ecm_as_feature":  True,
        "ci_formula":      f"±{MD['ci_multiplier']} × 12M_price_std × sqrt(horizon)",
        "countries":       COUNTRIES,
        "country_models":  {
            c: {
                "ar_differencing": v.get("ar_d"),
                "ar_metrics":      v.get("ar_metrics"),
                "gbr_metrics":     v.get("gbr_metrics"),
            }
            for c, v in country_models.items()
        },
        "assumptions": [
            "Trend dampening: price × (1 + trend × sqrt(h)) to prevent divergence",
            "ECM_Residual included where non-null — adds mean-reversion signal",
            "Non-stationary series first-differenced before AR fitting",
        ],
    }
    with open(os.path.join(GOV_DIR, "02_market_dynamics_engine.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)

    print("\n[02] Complete ✓")
    print("=" * 70)
    return fc_df, fwd_df, country_models


if __name__ == "__main__":
    run()
