# =============================================================================
# AATIP — utils/validators.py  |  Data integrity and leakage prevention
# =============================================================================
import os, sys
import pandas as pd
import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)


def check_temporal_split(train: pd.DataFrame, test: pd.DataFrame,
                          context: str = "") -> None:
    """Asserts zero overlap between train and test time periods."""
    train_years = set(train["Year"].unique())
    test_years  = set(test["Year"].unique())
    overlap = train_years & test_years
    if overlap:
        raise ValueError(
            f"TEMPORAL LEAKAGE [{context}]: train/test year overlap: {sorted(overlap)}"
        )
    if max(train_years) >= min(test_years):
        raise ValueError(
            f"TEMPORAL LEAKAGE [{context}]: train max year {max(train_years)} "
            f">= test min year {min(test_years)}"
        )
    print(f"  [OK] Temporal split [{context}]: "
          f"train {sorted(train_years)} | test {sorted(test_years)}")


def check_no_leakage(features: list, context: str = "") -> None:
    """Raises if any feature is in the leakage guard."""
    from config.features import LEAKAGE_GUARD
    hits = [f for f in features if f in LEAKAGE_GUARD]
    if hits:
        raise ValueError(f"LEAKAGE [{context}]: {hits} are in LEAKAGE_GUARD")


def safe_features(features: list, df_columns: list, context: str = "") -> list:
    """
    Returns the subset of features actually present in the DataFrame.
    Logs any missing ones as warnings (not errors — graceful degradation).
    """
    present = [f for f in features if f in df_columns]
    missing = [f for f in features if f not in df_columns]
    if missing:
        print(f"  [WARN] {context}: {len(missing)} features missing from data: {missing}")
    return present


def impute_train_only(df_train: pd.DataFrame,
                      df_test: pd.DataFrame,
                      features: list) -> tuple:
    """
    Imputes NaN values using column medians computed ONLY on train set.
    Critical: test set is imputed with train medians — no test data leaks into imputation.
    Returns (X_train_imputed, X_test_imputed, medians_series).
    """
    medians = df_train[features].median()
    X_train = df_train[features].fillna(medians)
    X_test  = df_test[features].fillna(medians) if len(df_test) > 0 else df_test[features]
    return X_train, X_test, medians


def check_mis_range(df: pd.DataFrame, context: str = "") -> None:
    """Warns if any MIS values fall outside the expected range."""
    lo, hi = CFG["mis"]["valid_range"]
    out_of_range = df[(df["MIS"] < lo) | (df["MIS"] > hi)]
    if len(out_of_range) > 0:
        print(f"  [WARN] {context}: {len(out_of_range)} MIS values outside [{lo}, {hi}]")
    else:
        print(f"  [OK]   MIS range [{context}]: all values in [{lo}, {hi}]")


def check_corridor_completeness(df: pd.DataFrame) -> None:
    """Reports coverage per corridor, flagging sparse years."""
    expected = CFG["corridors"]["all"]
    present  = df["Pair_ID"].unique().tolist()
    missing  = [c for c in expected if c not in present]
    if missing:
        print(f"  [WARN] Missing corridors: {missing}")
    for corridor in expected:
        sub = df[df["Pair_ID"] == corridor]
        sparse = {
            yr: ct
            for yr, ct in sub.groupby("Year").size().items()
            if ct < 12
        }
        if sparse:
            print(f"  [INFO] {corridor}: sparse years {sparse}")
    print(f"  [OK]   {len(present)}/{len(expected)} corridors present")
