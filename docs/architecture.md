# System Architecture

## Design Principles

AATIP is built around four non-negotiable architectural principles:

**1. No black boxes.** Every analytical layer has a documented, testable, auditable role. No single model is asked to do everything.

**2. Theory before ML.** The economic composite score in Engine 03 is always computed first. ML provides empirical refinement — it cannot override economic fundamentals.

**3. Temporal discipline.** No random splits. No data from the future in any training fold. TimeSeriesSplit exclusively. This is structural, not optional.

**4. Graceful degradation.** The platform is designed for imperfect African data conditions. Missing price data, delayed reporting, and partial coverage are expected — not exceptional cases.

---

## Nine-Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 9 │  DECISION DASHBOARD                                  │
│          │  4-page Streamlit: map · deep-dive · policy · alert  │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 8 │  POLICY SIMULATION ENGINE (E05)                      │
│          │  AfCFTA scenarios · elasticity sweep · cross-val     │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 7 │  ROUTE & COST INTELLIGENCE                           │
│          │  Hard gates · logistics · feasibility                │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 6 │  INFORMAL TRADE DETECTION (E04)                      │
│          │  IsolationForest · P(Informal) · UNCTAD calibration  │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 5 │  TRADE MATCHING ENGINE (E03)                         │
│          │  Lasso + economic composite · 3-layer hierarchy      │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 4 │  MARKET DYNAMICS ENGINE (E02)                        │
│          │  AR(p) + GBR · Forward MIS · price forecasts + CI   │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 3 │  SUPPLY / DEFICIT INTELLIGENCE (E01)                 │
│          │  Random Forest · deficit recall · early warning      │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 2 │  FEATURE ENGINEERING SUBSTRATE                       │
│          │  144 features · MIS · lags · climate · logistics     │
├──────────┼──────────────────────────────────────────────────────┤
│  Layer 1 │  DATA INTEGRATION                                    │
│          │  WFP VAM · FAOSTAT · UN Comtrade · ERA5 · LPI       │
└──────────┴──────────────────────────────────────────────────────┘
```

---

## Pairwise Intelligence

The platform analyzes **bilateral corridors**, not countries in isolation. Each corridor is its own economic system with unique:

- Transport distance and cost
- Border regime and friction  
- Seasonal complementarity
- Historical trade structure
- Informal trade propensity

This design gives corridor-level realism. A country-level analysis might identify that Tanzania holds a production advantage — a pairwise analysis specifies *which corridors* can monetize that advantage, at what logistics cost, with what informal suppression probability.

**Six corridors modelled:**

```
Kenya ────────────→ Tanzania      (KEN→TAN)
Kenya ────────────→ Zambia        (KEN→ZAM)
Tanzania ─────────→ Kenya         (TAN→KEN)  ← Primary corridor
Tanzania ─────────→ Zambia        (TAN→ZAM)
Zambia ───────────→ Kenya         (ZAM→KEN)
Zambia ───────────→ Tanzania      (ZAM→TAN)
```

---

## Trade Matching Hierarchy

The hierarchy in Engine 03 is **non-negotiable**. It reflects the epistemic priority of economic theory over empirical pattern-matching.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Economic Composite Score                          │
│  Always computed. Always interpretable.                     │
│  Grounds the score in theory before ML sees the data.       │
│                                                             │
│  Score = norm( 0.35×MIS_MA3 + 0.30×SupplyConf             │
│               + 0.20×RouteF  − 0.15×MktFric )              │
└────────────────────────┬────────────────────────────────────┘
                         │ feeds into
┌────────────────────────▼────────────────────────────────────┐
│  LAYER 2: Lasso ML Refinement                               │
│  Empirically optimises the ranking against historical data. │
│  26/37 features selected. α by TimeSeriesSplit CV.          │
│                                                             │
│  Final = 0.60×EconScore + 0.40×MLScore                     │
└────────────────────────┬────────────────────────────────────┘
                         │ subject to
┌────────────────────────▼────────────────────────────────────┐
│  LAYER 3: Hard Rule Gates                                   │
│  Binary feasibility constraints. Override everything.       │
│  Gate fails → Final score = 0, regardless of Layers 1+2.   │
│                                                             │
│  Gate = 1[RouteF ≥ 0.30] × 1[MktFric ≤ 0.80]              │
│  42% of observations blocked by at least one gate.          │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Execution

```
run_pipeline.py
     │
     ├─ Step 1: 01_supply_engine.py          → surplus_deficit_predictions.csv
     ├─ Step 2: 02_market_dynamics_engine.py → forward_mis.csv + price_forecasts.csv
     ├─ Step 3: 03_trade_matching_engine.py  → corridor_rankings.csv
     ├─ Step 4: 04_informal_trade_engine.py  → informal_trade_estimates.csv
     ├─ Step 5: 05_policy_simulation_engine.py → policy_scenarios.csv
     ├─ Step 6: 06_validation_econometrics.py → validation_report.json
     └─ Step 7: merge_all_outputs()          → AATIP_Final_Intelligence.csv
```

Each engine is loaded via `importlib` (handles numeric filename prefixes). A failed engine logs the error and the pipeline continues — no single point of failure.

---

## Governance Layer

```
config/
├── settings.yaml      ← Every parameter. One file. Single source of truth.
└── features.py        ← Feature registry + LEAKAGE_GUARD enforcement

governance/
├── assumptions.md     ← Economic assumptions, validation results, limitations
└── model_cards/
    ├── 01_supply_engine.json
    ├── 02_market_dynamics_engine.json
    ├── 03_trade_matching_engine.json
    ├── 04_informal_trade_engine.json
    ├── 05_policy_simulation_engine.json
    └── 06_validation_econometrics.json
```

**Leakage Guard:** 12 policy output columns are formally listed in `LEAKAGE_GUARD`. The validation layer raises a `ValueError` at runtime if any guarded column appears as a model input. This runs before the first training call — not as a post-hoc check.
