# Dataset Description

## Overview

The AATIP master dataset contains **532 corridor-month observations** across **6 bilateral trade pairs** and **144 engineered features**, spanning Kenya, Tanzania, and Zambia from 2013 to 2022.

```
Corridors:  KEN→TAN  |  KEN→ZAM  |  TAN→KEN
            TAN→ZAM  |  ZAM→KEN  |  ZAM→TAN

Temporal:   Training: 2013–2021  |  Test: 2022  |  Excluded: 2023 (2 rows/corridor)
Structure:  Unbalanced panel — KEN↔ZAM has observations from 2013;
            all 6 corridors have full bilateral coverage from 2016.
```

---

## Data Sources

### 1. WFP Vulnerability Analysis and Mapping (VAM)
- **Variables:** Retail and wholesale prices (USD/kg) by market and month
- **Coverage:** 2013–2022, monthly
- **Markets sampled:** Multiple markets per country per month; averaged to country-level
- **Access:** [VAM Resource Center](https://www.wfp.org/food-security/assessments)
- **Note:** This is the primary price series. The dataset's temporal coverage reflects VAM market monitoring expansion — early years have sparser price observations.

### 2. FAOSTAT
- **Variables:** Production (tonnes), area harvested (ha), yield (hg/ha)
- **Coverage:** 2013–2022, annual (interpolated to monthly where needed)
- **Commodity:** Maize and related staples
- **Access:** [FAOSTAT](https://www.fao.org/faostat/en/)

### 3. UN Comtrade
- **Variables:** Export/import net weight (kg), net trade value (USD)
- **Coverage:** 2013–2022; HS commodity codes for maize
- **Access:** [UN Comtrade](https://comtrade.un.org/)
- **Note:** Formal trade only. Informal flows are reconstructed by Engine 04.

### 4. ERA5 Reanalysis (ECMWF)
- **Variables:** Monthly precipitation (mm), mean temperature (°C)
- **Resolution:** 0.25° × 0.25° — averaged to country centroid
- **Coverage:** 2013–2022, monthly
- **Access:** [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)

### 5. World Bank LPI / AfDB
- **Variables:** Transport cost per kg, border crossing days, logistics friction index
- **Coverage:** Cross-sectional estimates, corridor-specific
- **Access:** [World Bank LPI](https://lpi.worldbank.org/), AfDB corridor assessments

---

## Feature Engineering

### Feature Groups (144 total)

| Group | Count | Key Features |
|---|---|---|
| Price Dynamics | 40 | MA3/MA6/MA12, volatility, z-score, MoM/YoY changes, lags |
| Supply & Production | 21 | Production, area, yield, net supply, surplus score |
| Climate & Harvest | 20 | Rainfall anomaly, drought index, heat stress, harvest calendar |
| MIS & Convergence | 16 | MIS, MIS_MA3/6/12, persistence, convergence speed, ECM residual |
| Logistics & Cost | 12 | Transport cost, border days, friction, landlocked flag |
| Trade Signals | 15 | Arbitrage signal/profit, informal trade signal, corridor reliability |
| Regional & Flags | 10 | EAC/SADC/AfCFTA flags, crisis signal, food security flag |
| Policy (Leakage Guard) | 9 | Pre-computed policy outputs — **never used as model inputs** |
| Econometric Outputs | 1 | Cointegration flag |

### The MIS Formula

The Market Inefficiency Score is computed per corridor-month:

```python
MIS = (Price_Partner - Price_Local - Transport_Cost) / Transport_Cost

# Where:
# Price_Partner  = wholesale price in importer country (USD/kg)
# Price_Local    = wholesale price in exporter country (USD/kg)
# Transport_Cost = total logistics cost, corridor-specific (USD/kg)
#                = border_friction_cost + transport_cost_per_km × distance
```

**MIS interpretation:**
- `MIS > 0` → trade is economically viable
- `MIS = 0` → exact arbitrage break-even
- `MIS < 0` → markets have converged; reverse trade would be loss-making

### Temporal Features

All temporal features are constructed to prevent leakage — only past information is used:

```python
MIS_Lag1    = MIS at t-1          # 1-month lag
MIS_Lag2    = MIS at t-2          # 2-month lag
MIS_MA3     = mean(MIS[t-2:t])    # 3-month trailing average
MIS_MA6     = mean(MIS[t-5:t])    # 6-month trailing average
MIS_MA12    = mean(MIS[t-11:t])   # 12-month trailing average
MIS_Zscore  = (MIS - μ) / σ       # standardized against corridor history
```

### Leakage Guard

12 columns encode future policy outcomes and are formally protected:

```python
LEAKAGE_GUARD = [
    "Policy_MIS_20pct_Border_Reduction",
    "Policy_MIS_10pct_Transport_Reduction",
    "Policy_MIS_AfCFTA_Full",
    "Policy_Uplift_S1", "Policy_Uplift_S2", "Policy_Uplift_AfCFTA",
    "Est_Trade_Increase_S1_pct", "Est_Trade_Increase_S2_pct",
    "Est_Trade_Increase_AfCFTA_pct",
    "Coint_Flag_36M", "Coint_Pval_36M",
    "Half_Life_Price_Reversion_M",
]
```

At pipeline start, `validate_registry()` raises a `ValueError` if any guarded column appears in any engine's feature list. This is structural, not optional.

---

## Temporal Structure

```
2013  2014  2015  2016  2017  2018  2019  2020  2021  |  2022  | 2023
─────────────────────────────────────────────────────────────────────
KEN↔ZAM: ████████████████████████████████████████████  │  ████  │  ██
TAN↔KEN: ────────────────█████████████████████████████  │  ████  │  ██
TAN↔ZAM: ────────────────█████████████████████████████  │  ████  │  ██
─────────────────────────────────────────────────────────────────────
                         ← TRAIN (2013-2021) →          │  TEST  │ excl.
```

- **2013–2015:** Only KEN↔ZAM corridors (WFP VAM coverage not yet full)
- **2016–2021:** All 6 corridors — full bilateral coverage (training set)
- **2022:** All 6 corridors — held out completely; never seen during training
- **2023:** Only 2 rows per corridor — excluded from all modelling

---

## Accessing the Data

The raw price data is governed by WFP VAM licensing terms. The master analytical CSV (`AATIP_Intelligence_Master_144_Feature_Claude.csv`) with all 144 engineered features is available on request for academic and research use.

To replicate feature engineering from raw sources:
1. Download WFP VAM prices for Kenya, Tanzania, Zambia (2013–2022)
2. Download FAOSTAT maize production data
3. Download UN Comtrade bilateral trade flows (HS code 1005)
4. Download ERA5 monthly precipitation and temperature
5. Apply logistics cost parameters from `config/settings.yaml`
6. Run the feature engineering scripts (forthcoming in v2.0)
