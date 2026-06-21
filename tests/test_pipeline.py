"""
AATIP — Test Suite
==================
Tests for feature registry validation, temporal leakage prevention,
MIS computation, and engine smoke tests.

Run: pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Allow imports from the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aatip.config.features import (
    validate_registry, LEAKAGE_GUARD,
    SUPPLY_FEATURES, SUPPLY_TARGET,
    TRADE_LASSO_FEATURES,
)
from aatip.utils.validators import (
    check_temporal_split, check_no_leakage,
    safe_features, impute_train_only,
)
from aatip.utils.metrics import mis_label


# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_df():
    """Minimal valid DataFrame with required columns."""
    n = 60
    pairs = ['TAN→KEN', 'ZAM→KEN']
    rows = []
    for pair in pairs:
        exporter = pair.split('→')[0]
        importer = pair.split('→')[1]
        for year in range(2016, 2022):
            for month in range(1, 7):
                rows.append({
                    'Pair_ID': pair, 'Exporter': exporter, 'Importer': importer,
                    'Year': year, 'Month': month,
                    'MIS': np.random.uniform(-2, 2),
                    'Exporter_Price_Wholesale_USD_kg': np.random.uniform(0.2, 0.5),
                    'Importer_Price_Wholesale_USD_kg': np.random.uniform(0.2, 0.5),
                    'Transport_Cost_USD_kg': 0.062,
                    'Border_Friction_Cost_USD_kg': 0.045,
                    'Price_Gap_USD_kg': np.random.uniform(-0.3, 0.3),
                    'Arbitrage_Signal': np.random.randint(0, 2),
                    'Route_Feasibility': np.random.uniform(0.3, 1.0),
                    'Market_Friction_Index': np.random.uniform(0.0, 0.8),
                    'Supply_Confidence': np.random.uniform(0.0, 1.0),
                    'MIS_MA3': np.random.uniform(-2, 2),
                    'Exporter_Surplus_Score': np.random.choice([-1, 0, 1]),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def train_test_dfs(minimal_df):
    train = minimal_df[minimal_df['Year'] <= 2020]
    test  = minimal_df[minimal_df['Year'] == 2021]
    return train, test


# ── FEATURE REGISTRY TESTS ───────────────────────────────────────────────────

class TestFeatureRegistry:

    def test_supply_features_all_valid_strings(self):
        assert all(isinstance(f, str) and len(f) > 0 for f in SUPPLY_FEATURES)

    def test_supply_target_is_string(self):
        assert isinstance(SUPPLY_TARGET, str)

    def test_leakage_guard_non_empty(self):
        assert len(LEAKAGE_GUARD) >= 9, "Leakage guard should protect at least 9 policy output columns"

    def test_validate_registry_with_all_present(self, minimal_df):
        # Add all needed columns for a clean pass
        full_cols = list(minimal_df.columns) + SUPPLY_FEATURES
        missing = validate_registry(full_cols, verbose=False)
        # Only SUPPLY_FEATURES are checked in minimal validation
        assert isinstance(missing, dict)

    def test_no_supply_feature_in_leakage_guard(self):
        violations = [f for f in SUPPLY_FEATURES if f in LEAKAGE_GUARD]
        assert len(violations) == 0, f"Supply features in leakage guard: {violations}"

    def test_no_trade_feature_in_leakage_guard(self):
        violations = [f for f in TRADE_LASSO_FEATURES if f in LEAKAGE_GUARD]
        assert len(violations) == 0, f"Trade features in leakage guard: {violations}"


# ── TEMPORAL VALIDATION TESTS ─────────────────────────────────────────────────

class TestTemporalValidation:

    def test_clean_split_passes(self, train_test_dfs):
        train, test = train_test_dfs
        # Should not raise
        check_temporal_split(train, test, context="test_clean_split")

    def test_overlapping_years_raises(self, minimal_df):
        train = minimal_df[minimal_df['Year'] <= 2020]
        test  = minimal_df[minimal_df['Year'] == 2020]  # overlap!
        with pytest.raises(ValueError, match="TEMPORAL LEAKAGE"):
            check_temporal_split(train, test, context="test_overlap")

    def test_no_leakage_passes_clean_features(self):
        clean = ['MIS', 'Supply_Confidence', 'Route_Feasibility']
        check_no_leakage(clean, context="test_clean")  # should not raise

    def test_no_leakage_raises_on_policy_column(self):
        dirty = ['MIS', 'Policy_MIS_20pct_Border_Reduction']
        with pytest.raises(ValueError, match="LEAKAGE"):
            check_no_leakage(dirty, context="test_dirty")

    def test_safe_features_returns_only_present(self, minimal_df):
        wanted = ['MIS', 'Route_Feasibility', 'NONEXISTENT_COLUMN']
        result = safe_features(wanted, minimal_df.columns.tolist(), context="test")
        assert 'MIS' in result
        assert 'Route_Feasibility' in result
        assert 'NONEXISTENT_COLUMN' not in result


# ── IMPUTATION TESTS ─────────────────────────────────────────────────────────

class TestImputation:

    def test_train_only_imputation(self, train_test_dfs):
        train, test = train_test_dfs
        features = ['MIS', 'Route_Feasibility', 'Supply_Confidence']

        # Add some NaN to test
        train_c = train[features].copy()
        test_c  = test[features].copy()
        train_c.iloc[0, 0] = np.nan
        test_c.iloc[0, 1]  = np.nan

        X_tr, X_te, medians = impute_train_only(
            train_c.assign(**{k: train_c[k] for k in features}),
            test_c.assign(**{k: test_c[k] for k in features}),
            features
        )
        assert X_tr.isnull().sum().sum() == 0
        assert X_te.isnull().sum().sum() == 0

    def test_medians_from_train_only(self, train_test_dfs):
        train, test = train_test_dfs
        features = ['MIS']
        X_tr, X_te, medians = impute_train_only(train, test, features)
        train_median = train['MIS'].median()
        assert abs(medians['MIS'] - train_median) < 1e-10


# ── MIS COMPUTATION TESTS ─────────────────────────────────────────────────────

class TestMISLogic:

    def test_mis_positive_when_gap_exceeds_logistics(self):
        price_partner  = 0.45
        price_local    = 0.25
        logistics_cost = 0.062 + 0.045
        mis = (price_partner - price_local - logistics_cost) / logistics_cost
        assert mis > 0

    def test_mis_negative_when_gap_below_logistics(self):
        price_partner  = 0.30
        price_local    = 0.28
        logistics_cost = 0.062 + 0.045
        mis = (price_partner - price_local - logistics_cost) / logistics_cost
        assert mis < 0

    def test_mis_zero_at_exact_breakeven(self):
        logistics = 0.10
        price_diff = logistics  # exact breakeven
        mis = (price_diff - logistics) / logistics
        assert abs(mis) < 1e-10

    def test_mis_label_crisis(self):
        result = mis_label(2.0)
        assert result['severity'] == 'critical'

    def test_mis_label_converged(self):
        result = mis_label(-0.5)
        assert result['severity'] == 'none'

    def test_mis_label_arbitrage(self):
        result = mis_label(0.3)
        assert result['severity'] == 'moderate'

    def test_mis_label_nan_handled(self):
        result = mis_label(None)
        assert result['label'] == 'Unknown'


# ── POLICY SIMULATION TESTS ───────────────────────────────────────────────────

class TestPolicyMechanics:

    def test_border_reduction_lowers_logistics(self):
        original_border = 0.045
        reduction = 0.20
        new_border = original_border * (1 - reduction)
        assert new_border < original_border

    def test_mis_increases_when_logistics_fall(self):
        gap = 0.15
        logistics_orig = 0.107
        logistics_new  = logistics_orig * 0.80

        mis_orig = (gap - logistics_orig) / logistics_orig
        mis_new  = (gap - logistics_new)  / logistics_new
        assert mis_new > mis_orig

    def test_trade_increase_proportional_to_elasticity(self):
        logistics_orig = 0.107
        logistics_new  = 0.080
        delta_L_pct    = abs((logistics_new - logistics_orig) / logistics_orig)

        e1, e2 = 1.0, 2.0
        delta_trade1 = e1 * delta_L_pct
        delta_trade2 = e2 * delta_L_pct
        assert abs(delta_trade2 / delta_trade1 - 2.0) < 1e-10

    def test_elasticity_range_covered(self):
        """Sensitivity sweep should cover min to max elasticity."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'aatip', 'config', 'settings.yaml'
        )
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            ps = cfg['policy_simulation']
            assert ps['elasticity_min'] < ps['elasticity_base'] < ps['elasticity_max']


# ── INTEGRATION SMOKE TEST ────────────────────────────────────────────────────

class TestSmoke:

    def test_engine_01_imports_cleanly(self):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "e01",
                os.path.join(os.path.dirname(__file__),
                             '..', 'aatip', 'models', '01_supply_engine.py')
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert hasattr(mod, 'run')
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_settings_yaml_loads(self):
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'aatip', 'config', 'settings.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("settings.yaml not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert 'mis' in cfg
        assert 'temporal' in cfg
        assert 'corridors' in cfg

    def test_features_py_loads(self):
        from aatip.config.features import (
            SUPPLY_FEATURES, SUPPLY_TARGET, TRADE_LASSO_FEATURES,
            LEAKAGE_GUARD, validate_registry
        )
        assert len(SUPPLY_FEATURES) > 0
        assert len(TRADE_LASSO_FEATURES) > 0
        assert len(LEAKAGE_GUARD) > 0
        assert callable(validate_registry)
