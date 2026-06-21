# Installation & Usage Guide

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| scikit-learn | ≥ 1.4.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| scipy | ≥ 1.10.0 |
| matplotlib | ≥ 3.7.0 |
| pyyaml | ≥ 6.0.0 |
| streamlit *(dashboard)* | ≥ 1.30.0 |
| plotly *(dashboard)* | ≥ 5.18.0 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/aatip.git
cd aatip
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Setup

The platform expects the master analytical dataset at:

```
aatip/AATIP_Intelligence_Master_144_Feature_Claude.csv
```

This file contains 532 rows × 144 engineered features across 6 bilateral corridors (2013–2022). See [docs/dataset.md](dataset.md) for the full feature description.

*Raw WFP VAM price data is subject to licensing — contact the author for the pre-processed analytical CSV.*

---

## Running the Pipeline

### Full pipeline (recommended)

Runs all 6 engines in sequence, then merges outputs:

```bash
python aatip/pipeline/run_pipeline.py
```

Expected output:
```
═══════════════════════════════════════════════════
  AATIP — FULL PIPELINE EXECUTION
═══════════════════════════════════════════════════
  01  SUCCESS  (5.0s)
  02  SUCCESS  (0.7s)
  03  SUCCESS  (0.2s)
  04  SUCCESS  (0.2s)
  05  SUCCESS  (0.2s)
  06  SUCCESS  (0.2s)
  merge  SUCCESS
  7/7 steps successful
```

### Individual engines

Each engine can be run independently:

```bash
python aatip/models/01_supply_engine.py
python aatip/models/02_market_dynamics_engine.py
python aatip/models/03_trade_matching_engine.py
python aatip/models/04_informal_trade_engine.py
python aatip/models/05_policy_simulation_engine.py
python aatip/models/06_validation_econometrics.py
```

---

## Dashboard

```bash
streamlit run aatip/dashboard/app.py
```

Opens at `http://localhost:8501` with four pages:

| Page | Content |
|---|---|
| **Trade Opportunity Map** | Corridor map with MIS-coded arcs, KPI metrics, ranking table |
| **Corridor Deep-Dive** | MIS time series, price gap, informal trade probability, price forecasts |
| **Policy Simulator** | Live sliders: border reduction, transport reduction, elasticity → recomputes in real time |
| **Early Warning Panel** | Crisis signals, MIS heatmap, deficit severity, pre-harvest calendar |

---

## Configuration

All parameters are in `aatip/config/settings.yaml`. Key settings:

```yaml
# MIS thresholds
mis:
  arbitrage_threshold: 0.0      # MIS > 0 → viable trade
  strong_arbitrage:    0.5      # MIS > 0.5 → high priority
  crisis_threshold:    1.5      # MIS > 1.5 → crisis alert

# Temporal split
temporal:
  train_years: [2013,2014,2015,2016,2017,2018,2019,2020,2021]
  test_years:  [2022]

# Trade matching weights (Engine 03)
trade_matching:
  w_mis_ma3:          0.35
  w_supply_confidence: 0.30
  w_route_feasibility: 0.20
  w_market_friction:   0.15

# Policy scenarios (Engine 05)
policy_simulation:
  elasticity_base: 2.0
  scenarios:
    AfCFTA_Phase1:
      border_reduction:    0.20
      transport_reduction: 0.00
    AfCFTA_Full:
      border_reduction:    0.50
      transport_reduction: 0.10
```

---

## Outputs

After a successful pipeline run:

```
outputs/
├── predictions/
│   ├── surplus_deficit_predictions.csv    # E01: country-month surplus/deficit
│   ├── forward_mis.csv                   # E02: 6-month forward MIS per corridor
│   ├── price_forecasts.csv               # E02: price forecasts with 90% CI
│   ├── corridor_rankings.csv             # E03: corridor rankings with scores
│   ├── informal_trade_estimates.csv      # E04: P(Informal) and volume estimates
│   └── policy_scenarios.csv             # E05: policy scenario results
│
├── reports/
│   ├── policy_headline_numbers.json      # Top-line policy impact numbers
│   ├── validation_report.json            # Full validation results
│   ├── econometric_summary.csv           # Cointegration, Granger, ECM, beta
│   ├── lasso_coefficients.csv            # Engine 03 feature coefficients
│   ├── policy_sensitivity.csv            # Elasticity sensitivity table
│   └── pipeline_log.json                 # Per-engine status and timing
│
└── model_artifacts/
    ├── supply_engine_rf.pkl              # Fitted Random Forest
    ├── supply_engine_gb.pkl              # Fitted Gradient Boosting
    ├── market_dynamics_models.pkl        # AR + GBR per country
    └── trade_matching_lasso.pkl          # Fitted Lasso pipeline

AATIP_Final_Intelligence.csv              # All outputs merged: 532 rows × 188 cols
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'`**
```bash
pip install pyyaml
```

**`FileNotFoundError: AATIP_Intelligence_Master_144_Feature_Claude.csv`**
Place the master CSV at the path specified in `config/settings.yaml`:
```yaml
data:
  master_csv: "AATIP_Intelligence_Master_144_Feature_Claude.csv"
```

**Pipeline engine fails with `KeyError`**
Check that your dataset contains the expected columns. Run the feature registry validator:
```python
from aatip.config.features import validate_registry
import pandas as pd
df = pd.read_csv("your_data.csv")
validate_registry(df.columns.tolist(), verbose=True)
```

**Dashboard shows no data**
The dashboard requires the pipeline to have been run first. Check `outputs/predictions/` exists and contains CSV files.

---

## Running Tests

```bash
pytest tests/ -v
```

*Test suite covers feature registry validation, temporal leakage checks, and basic engine smoke tests.*
