# Methodology Reference

Quick reference for all analytical methods. Full derivations in the [Technical Appendix](../AATIP_Technical_Appendix.docx).

---

## Core Metric: Market Inefficiency Score

```
MIS(i,j,t) = [P(j,t) - P(i,t) - L(i,j)] / L(i,j)

Logistics: L(i,j) = Transport_Cost(i,j) + Border_Friction_Cost(i,j)
CI (90%):  P̂(t+h) ± 1.64 × σ₁₂ × √h
```

**Why normalize by logistics costs, not price level?**  
Scale invariance: MIS = 0.5 means the price gap exceeds logistics costs by 50%, whether the commodity costs $0.25/kg or $2.50/kg. This makes cross-corridor comparison valid and policy simulation tractable.

---

## Engine 01: Supply & Deficit Classifier

**Model:** Random Forest (Breiman, 2001)  
**Target:** Surplus/Neutral/Deficit (−1, 0, 1) per country-month

```python
# Class probabilities
P(class k | x) = (1/B) × Σ 1[T_b(x) = k]     # B = 200 trees

# Deficit warning flag
Deficit_Warning = 1 if P(deficit | x) ≥ 0.45
```

**Key design choices:**
- `class_weight='balanced'` — corrects class imbalance without oversampling
- Priority metric: **deficit recall** — false negatives cost more than false positives
- Temporal CV only: `TimeSeriesSplit(n_splits=5)`

**Test results (2022 held-out):**

| Metric | Value | Threshold |
|---|---|---|
| Deficit Recall | 1.000 | ≥ 0.65 |
| Macro F1 | 1.000 | ≥ 0.50 |
| Brier Score (CV) | 0.020 | < 0.150 |

---

## Engine 02: Market Dynamics Engine

**Primary output:** Forward MIS (6-month horizon)  
**Secondary output:** Price forecasts with 90% CI

### Autoregressive Baseline
```
P(t+h) = α + β₁P(t-1) + β₂P(t-2) + β₃P(t-3) + β₆P(t-6) + β₁₂P(t-12) + ε
```
Series first-differenced if ADF test (t < −2.86) rejects stationarity.

### Gradient Boosting Enrichment
```
F_m(x) = F_{m-1}(x) + η × h_m(x)      # η = 0.05, M = 200
```
Features include: climate leading indicators, supply shocks, ECM residuals, harvest calendar.

### Forward MIS Projection
```
FwdMIS(h) = [P̂_j(t+h) - P̂_i(t+h) - L(i,j)] / L(i,j)

Trend dampening: P̂(t+h) = P(t) × (1 + trend × √h)
```
`√h` dampening prevents linear extrapolation from diverging at longer horizons.

---

## Engine 03: Trade Matching Engine

### Layer 1: Economic Composite
```
EconScore = norm(0.35×MIS_MA3 + 0.30×SupplyConf + 0.20×RouteF − 0.15×MktFric)
```
All inputs normalized to [0,1] before weighting. Weights from World Bank (2020) SSA elasticity estimates.

### Layer 2: Lasso Regression
```
β̂ = argmin { ||log(1+y) - Xβ||² + α||β||₁ }
```
- Target: `log(1 + Exporter_Export_tonnes)`
- α selected by `TimeSeriesSplit` cross-validation
- 26/37 features non-zero after regularization

### Layer 3: Hard Gates
```
Gate = 1[RouteF ≥ 0.30] × 1[MktFric ≤ 0.80]
Final = (0.60×EconScore + 0.40×MLScore) × Gate
```

**Test result:** Spearman ρ = 0.772 on 2022 held-out data.

---

## Engine 04: Informal Trade Engine

```
P(Informal) = 0.40 × P_MIS + 0.30 × P_Persist + 0.30 × P_Anomaly

P_MIS     = σ(2 × MIS_Zscore)                    # sigmoid
P_Persist = min(Persistence_Months / 12, 1.0)     # normalized
P_Anomaly = IsolationForest.score_samples(X)       # normalized to [0,1]

Volume (mid) = Formal_Vol × P_Informal × 3.0       # UNCTAD: 2-4× formal
```

**IsolationForest:** Liu, Ting & Zhou (2008). Identifies anomalous price-trade decoupling without labeled training data — appropriate since informal trade has no ground truth labels.

---

## Engine 05: Policy Simulation Engine

```
# Post-reform logistics costs
L_new = T × (1 - τ_T) + B × (1 - τ_B)

# Simulated MIS
MIS_new = (PriceGap - L_new) / L_new

# Trade volume response  
ΔTrade% = ε × |ΔL%|    where ΔL% = (L_new - L) / L

# Incremental volume
ΔVol = Formal_Vol × ΔTrade%
```

**Scenarios:**

| Scenario | τ_Border | τ_Transport | AU Target |
|---|---|---|---|
| Baseline | 0% | 0% | — |
| AfCFTA Phase 1 | 20% | 0% | 2027 single-window |
| AfCFTA Full | 50% | 10% | 2035 full implementation |

**Elasticity range:** ε = 1.0–3.5 (World Bank 2020 SSA estimates)

---

## Engine 06: Econometric Validation

### Engle-Granger Cointegration (1987)

```
# Step 1: Cointegrating regression
P_j(t) = α + β × P_i(t) + ê(t)

# Step 2: ADF test on residuals
Δê(t) = γ × ê(t-1) + u(t)
# γ < 0 and t(γ) < -2.86 → cointegrated
```

### Granger Causality F-Test (Granger, 1969)

```
F = [(RSS_R - RSS_U) / q] / [RSS_U / (n-k)]
# RSS_R: restricted (importer lags only)
# RSS_U: unrestricted (+ exporter lags)
# q: number of restrictions (lag order = 3)
```

### Price Transmission Beta

```
ΔP_j(t) = α + β × ΔP_i(t) + ε(t)    [OLS on first differences]

# β = 1.0 → full pass-through
# β < 1.0 → partial transmission
# β > 1.0 → amplification
```

### ECM Half-Life

```
Half_Life = -ln(2) / γ    [months]
# γ from ADF regression on ECM residuals
# Shorter = faster market adjustment
```

---

## Validation Protocol

**Non-negotiable temporal structure:**

```
Year:  2013 2014 2015 2016 2017 2018 2019 2020 2021 │ 2022
       ──────────────────────────────────────────────┼──────
Train: ████ ████ ████ ████ ████ ████ ████ ████ ████ │
Test:                                                 │ ████
CV:    TimeSeriesSplit(n_splits=5) — expanding window only
```

Every reported metric is from **data the model never saw during training.** The pipeline validates this before running — year overlap in any CV fold raises a `ValueError`.

---

## References

- Baulch, B. (1997). Transfer costs, spatial arbitrage, and testing for food market integration. *American Journal of Agricultural Economics, 79*(2), 477–487.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica, 55*(2), 251–276.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models. *Econometrica, 37*(3), 424–438.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *IEEE ICDM*, 413–422.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B, 58*(1), 267–288.
- World Bank. (2020). *The African Continental Free Trade Area: Economic and distributional effects.*
