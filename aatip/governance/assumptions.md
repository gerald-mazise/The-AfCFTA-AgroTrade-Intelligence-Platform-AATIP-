# AATIP — Governance & Assumptions

## Platform Identity

AATIP is not a price prediction tool. It is an **agricultural market transmission and coordination intelligence system** for the African Continental Free Trade Area.

Central intelligence thesis: *If markets function efficiently, prices converge after accounting for transport and friction costs. When they do not, AATIP diagnoses why.*

---

## MIS Formula

```
MIS = (Price_Partner - Price_Local - Transport_Cost) / Transport_Cost
```

| MIS Value | Interpretation | System Response |
|---|---|---|
| > 1.5 | Structural market failure | Critical alert |
| 0.5 – 1.5 | Strong arbitrage signal | High priority corridor |
| 0.0 – 0.5 | Arbitrage opportunity | Monitor and recommend |
| −0.2 – 0.0 | Near convergence | Normal monitoring |
| < −0.2 | Efficiently converged | No intervention needed |

---

## Economic Assumptions

| Parameter | Value | Justification |
|---|---|---|
| Trade elasticity (base) | 2.0 | World Bank SSA staple grain estimates |
| Trade elasticity (range) | 1.0 – 3.5 | Sensitivity sweep to bound uncertainty |
| Informal/formal ratio | 2× – 4× | UNCTAD EAC benchmark for staple grains |
| AfCFTA Phase 1 border reduction | 20% | AU 2027 interim target (single-window customs) |
| AfCFTA Full border reduction | 50% | AU 2035 full implementation target |
| AfCFTA Full transport reduction | 10% | Infrastructure co-investment assumed |
| ECM half-life ceiling | 24 months | Above this = structurally fragmented market |

---

## Data Assumptions

- **Price data**: WFP VAM retail and wholesale prices. Coverage is uneven — 2013–2015 only KEN↔ZAM corridors; full 6-corridor coverage from 2016.
- **Trade volumes**: UN COMTRADE-derived. Formal trade only. Informal trade is inferred via Engine 04.
- **Climate data**: Monthly precipitation/temperature at country centroid. Sub-national variation not captured.
- **Temporal coverage**: 2013–2022, partial years in early and late period.

## Temporal Split Rationale

- **Train**: 2013–2021 (all available history)
- **Test**: 2022 (held out; simulates real deployment on sparse recent data)
- **Excluded**: 2023 (only 2 rows per corridor — statistically insufficient)
- **CV**: TimeSeriesSplit, n=5, expanding window only. No random splits anywhere.

---

## Model Hierarchy — Trade Matching

Order is **fixed** — not co-equal:

1. **Economic composite** (floor): theory-grounded, always computed, always interpretable
2. **Lasso ML** (ranking): optimises against historical trade volumes
3. **Hard rule gates** (filters): binary feasibility constraints that override everything

---

## Validated Results Summary

| Engine | Primary Metric | Value | Threshold | Status |
|---|---|---|---|---|
| 01 Supply Classifier | Deficit Recall (test) | 1.000 | ≥ 0.65 | ✅ PASS |
| 01 Supply Classifier | Macro F1 (test) | 1.000 | ≥ 0.50 | ✅ PASS |
| 02 Market Dynamics | MAPE Kenya (val) | 0.104 | ≤ 0.20 | ✅ PASS |
| 02 Market Dynamics | MAPE Tanzania (test) | 0.084 | ≤ 0.20 | ✅ PASS |
| 03 Trade Matching | Spearman ρ (test) | 0.772 | ≥ 0.35 | ✅ PASS |
| 05 Policy Simulation | AfCFTA_Phase1 cross-val corr | 1.000 | ≥ 0.85 | ✅ PASS |
| 05 Policy Simulation | AfCFTA_Full cross-val corr | 1.000 | ≥ 0.85 | ✅ PASS |
| 06 Econometrics | Valid ECM residuals | 4/6 corridors | — | ℹ️ INFO |

---

## Policy Simulation Headline Results (base elasticity = 2.0)

| Scenario | Incremental Volume | Incremental Value | Months Unlocked |
|---|---|---|---|
| Baseline | 0 | $0 | 0 |
| AfCFTA Phase 1 (-20% border) | 9,227,698 tonnes | $2.8B | 17 |
| AfCFTA Full (-50% border, -10% transport) | 30,313,594 tonnes | $9.3B | 62 |

Sensitivity range at full implementation: **15M – 53M tonnes** (elasticity 1.0–3.5)

---

## Known Limitations

1. Informal trade volumes are **inferred** signals, not measurements.
2. Policy simulation uses **constant elasticity** — real response varies.
3. Climate signals are at **country level** — local shocks may not register.
4. Cointegration and Granger tests can only be run on corridors with **≥30 matched price observations** — KEN↔ZAM, TAN↔ZAM, ZAM↔TAN corridors are data-limited.
5. This is **partial equilibrium** modelling — general equilibrium effects (price level changes from large-scale reforms) are not modelled.
6. No causal identification: econometric tests establish statistical relationships, not experimental causation.
