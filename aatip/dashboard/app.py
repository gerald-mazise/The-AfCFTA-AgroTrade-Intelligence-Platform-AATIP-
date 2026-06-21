# =============================================================================
# AATIP — dashboard/app.py  |  Streamlit Dashboard
# Run: cd /home/claude/aatip && streamlit run dashboard/app.py
# =============================================================================

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

CORRIDORS  = CFG["corridors"]["all"]
COUNTRIES  = CFG["corridors"]["countries"]
MIS_CFG    = CFG["mis"]
COORDS     = CFG["country_coords"]
PRED_DIR   = os.path.join(BASE_DIR, CFG["data"]["predictions_dir"])
REP_DIR    = os.path.join(BASE_DIR, CFG["data"]["reports_dir"])


# ---------------------------------------------------------------------------
# MIS COLOUR HELPER
# ---------------------------------------------------------------------------
def mis_color(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#999999"
    if v > 1.5:  return "#d62728"
    if v > 0.5:  return "#e07b00"
    if v > 0.0:  return "#f4a100"
    if v > -0.2: return "#2ca02c"
    return "#1f77b4"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_all():
    final = os.path.join(BASE_DIR, "AATIP_Final_Intelligence.csv")
    master = os.path.join(BASE_DIR, CFG["data"]["master_csv"])
    df = pd.read_csv(final if os.path.exists(final) else master)

    outputs = {}
    for fname in ["price_forecasts","forward_mis","corridor_rankings",
                  "informal_trade_estimates","policy_scenarios"]:
        p = os.path.join(PRED_DIR, f"{fname}.csv")
        if os.path.exists(p):
            outputs[fname] = pd.read_csv(p)

    headlines = {}
    hl = os.path.join(REP_DIR, "policy_headline_numbers.json")
    if os.path.exists(hl):
        with open(hl) as f:
            headlines = json.load(f)

    return df, outputs, headlines


# ---------------------------------------------------------------------------
# PAGE 1 — TRADE OPPORTUNITY MAP
# ---------------------------------------------------------------------------
def page_map(df, outputs):
    st.title("🌍 Trade Opportunity Map")
    st.caption(
        "MIS > 0 means trade from exporter to importer is economically viable. "
        "Arc width = trade score. Arc colour = market efficiency."
    )

    years = sorted(df["Year"].unique())
    c1, c2 = st.columns([3, 1])
    with c2:
        yr  = st.selectbox("Year",  years, index=len(years)-1)
        mo  = st.selectbox("Month", sorted(df[df["Year"]==yr]["Month"].unique()),
                           index=len(df[df["Year"]==yr]["Month"].unique())-1)

    period = df[(df["Year"]==yr) & (df["Month"]==mo)].copy()

    # ── Metric bar ──────────────────────────────────────────────────────
    arb = int(period.get("Arbitrage_Signal", pd.Series(0)).sum()) if "Arbitrage_Signal" in period.columns else 0
    crs = int(period.get("Crisis_Signal", pd.Series(0)).sum()) if "Crisis_Signal" in period.columns else 0
    fsr = int(period.get("Food_Security_Risk_Flag", pd.Series(0)).sum()) if "Food_Security_Risk_Flag" in period.columns else 0
    avg_mis = round(float(period["MIS"].mean()), 3) if "MIS" in period.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Arbitrage Corridors", arb)
    m2.metric("Crisis Signals", crs)
    m3.metric("Food Security Flags", fsr)
    m4.metric("Avg MIS", avg_mis)
    st.divider()

    # ── Map ─────────────────────────────────────────────────────────────
    with c1:
        fig = go.Figure()

        # Corridor arcs
        for _, row in period.iterrows():
            exp, imp = row["Exporter"], row["Importer"]
            if exp not in COORDS or imp not in COORDS:
                continue
            mis_v = float(row.get("MIS", 0) or 0)
            ts    = float(row.get("Trade_Score_Final", row.get("Trade_Score", 0.5)) or 0.5)
            fig.add_trace(go.Scattergeo(
                lat=[COORDS[exp]["lat"], COORDS[imp]["lat"]],
                lon=[COORDS[exp]["lon"], COORDS[imp]["lon"]],
                mode="lines",
                line=dict(width=max(1, ts*7), color=mis_color(mis_v)),
                opacity=0.85,
                name=f"{row['Pair_ID']} (MIS={mis_v:.2f})",
                showlegend=True,
            ))

        # Country nodes
        for country, coords in COORDS.items():
            sup_rows = period[period["Exporter"]==country]
            surplus = int(sup_rows["Exporter_Surplus_Score"].mode()[0]) if (
                len(sup_rows) > 0 and "Exporter_Surplus_Score" in sup_rows.columns
            ) else 0
            nc = "#2ca02c" if surplus==1 else "#d62728" if surplus==-1 else "#aaaaaa"
            lbl = {1:"Surplus",0:"Neutral",-1:"Deficit"}.get(surplus,"?")
            fig.add_trace(go.Scattergeo(
                lat=[coords["lat"]], lon=[coords["lon"]],
                mode="markers+text",
                marker=dict(size=20, color=nc, line=dict(width=2,color="white")),
                text=[f"<b>{country}</b><br>{lbl}"],
                textposition="top center",
                name=country, showlegend=False,
            ))

        fig.update_layout(
            geo=dict(
                scope="africa",
                showland=True, landcolor="rgb(245,245,240)",
                showocean=True, oceancolor="rgb(220,235,250)",
                showcountries=True, countrycolor="rgb(200,200,200)",
                center=dict(lat=-5.0, lon=28.0),
                projection_scale=4.5,
            ),
            height=480, margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(orientation="h", y=-0.08),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Corridor table ─────────────────────────────────────────────────
    st.subheader("Corridor Rankings")
    rank_cols = ["Pair_ID","MIS","MIS_MA3","Trade_Score_Final","Corridor_Rank",
                 "Gate_Pass","Arbitrage_Signal","Arbitrage_Profit_USD_kg",
                 "Route_Feasibility","Food_Security_Risk_Flag"]
    avail = [c for c in rank_cols if c in period.columns]
    if avail:
        rank_df = period[avail].sort_values("Corridor_Rank" if "Corridor_Rank" in avail else avail[0])
        fmt = {c: "{:.3f}" for c in ["MIS","MIS_MA3","Trade_Score_Final",
                                       "Route_Feasibility","Arbitrage_Profit_USD_kg"]
               if c in rank_df.columns}
        st.dataframe(
            rank_df.style
            .background_gradient(subset=["MIS"] if "MIS" in rank_df.columns else [],
                                  cmap="RdYlGn", vmin=-2, vmax=2)
            .format(fmt, na_rep="—"),
            use_container_width=True, height=260
        )


# ---------------------------------------------------------------------------
# PAGE 2 — CORRIDOR DEEP-DIVE
# ---------------------------------------------------------------------------
def page_corridor(df, outputs):
    st.title("📈 Corridor Deep-Dive")
    corridor = st.selectbox("Corridor", CORRIDORS,
                             index=CORRIDORS.index(CFG["dashboard"]["default_corridor"]))

    sub = df[df["Pair_ID"]==corridor].sort_values(["Year","Month"]).copy()
    sub["Date"] = pd.to_datetime({"year":sub["Year"],"month":sub["Month"],"day":1})

    if len(sub)==0:
        st.warning("No data."); return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean MIS", f"{sub['MIS'].mean():.3f}")
    c2.metric("Arbitrage Months",
              int(sub["Arbitrage_Signal"].sum()) if "Arbitrage_Signal" in sub.columns else "—")
    c3.metric("Max Arb Profit",
              f"${sub['Arbitrage_Profit_USD_kg'].max():.4f}/kg" if "Arbitrage_Profit_USD_kg" in sub.columns else "—")
    c4.metric("P_Informal Mean",
              f"{sub['P_Informal'].mean():.3f}" if "P_Informal" in sub.columns else "—")
    st.divider()

    # MIS time series
    st.subheader("Market Inefficiency Score")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MIS"], name="MIS",
                              line=dict(color="#1f77b4",width=2)))
    if "MIS_MA3" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MIS_MA3"], name="MIS 3M MA",
                                  line=dict(color="#ff7f0e",width=2,dash="dash")))
    fig.add_hline(y=0, line_color="red", line_dash="dot", annotation_text="Arb threshold")
    fig.add_hline(y=0.5, line_color="orange", line_dash="dot", annotation_text="Strong signal")
    fig.update_layout(height=320, hovermode="x unified",
                       yaxis_title="MIS", xaxis_title="",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # Price gap + arbitrage profit
    if "Price_Gap_USD_kg" in sub.columns:
        st.subheader("Price Gap & Arbitrage Profit")
        fig2 = make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(
            go.Scatter(x=sub["Date"], y=sub["Price_Gap_USD_kg"],
                       name="Price Gap (USD/kg)", line=dict(color="#9467bd")),
            secondary_y=False)
        if "Arbitrage_Profit_USD_kg" in sub.columns:
            fig2.add_trace(
                go.Bar(x=sub["Date"], y=sub["Arbitrage_Profit_USD_kg"].clip(lower=0),
                       name="Arb Profit", marker_color="rgba(44,160,44,0.5)"),
                secondary_y=True)
        fig2.update_layout(height=280, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    # Informal trade probability
    if "P_Informal" in sub.columns:
        st.subheader("Informal Trade Probability")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sub["Date"], y=sub["P_Informal"],
                                   fill="tozeroy", name="P(Informal)",
                                   line=dict(color="#8c564b"),
                                   fillcolor="rgba(140,86,75,0.18)"))
        fig3.add_hline(y=CFG["informal_trade"]["high_confidence"],
                        line_color="#8c564b", line_dash="dot",
                        annotation_text="High confidence threshold")
        fig3.update_layout(height=240, yaxis_range=[0,1], yaxis_title="Probability")
        st.plotly_chart(fig3, use_container_width=True)

    # Price forecast
    if "price_forecasts" in outputs:
        exporter = sub["Exporter"].iloc[0]
        fc = outputs["price_forecasts"]
        fc_c = fc[fc["Country"]==exporter].sort_values("Horizon_M")
        if len(fc_c) > 0:
            st.subheader(f"Price Forecast — {exporter} (next {len(fc_c)} months)")
            fig4 = go.Figure()
            if "Exporter_Price_Wholesale_USD_kg" in sub.columns:
                fig4.add_trace(go.Scatter(
                    x=sub["Date"].tail(30), y=sub["Exporter_Price_Wholesale_USD_kg"].tail(30),
                    name="Historical", line=dict(color="#1f77b4")))
            fig4.add_trace(go.Scatter(
                x=list(range(1, len(fc_c)+1)), y=fc_c["Price_Forecast"].tolist(),
                name="Forecast", line=dict(color="#ff7f0e",dash="dash")))
            # CI band
            x_band = list(range(1,len(fc_c)+1)) + list(range(len(fc_c),0,-1))
            y_band = fc_c["CI_Upper"].tolist() + fc_c["CI_Lower"].tolist()[::-1]
            fig4.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself",
                                       fillcolor="rgba(255,127,14,0.15)",
                                       line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
            fig4.update_layout(height=270, xaxis_title="Months ahead", yaxis_title="USD/kg")
            st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3 — POLICY SIMULATOR
# ---------------------------------------------------------------------------
def page_policy(df, headlines):
    st.title("⚙️ Policy Simulator")
    st.markdown(
        "Simulate AfCFTA interventions. Sliders recompute MIS and trade volume "
        "estimates live. Pre-computed scenarios validated at corr=1.000 against "
        "dataset policy columns."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        b_red = st.slider("Border friction reduction (%)", 0, 60, 20, 5) / 100
    with c2:
        t_red = st.slider("Transport cost reduction (%)", 0, 20, 0, 2) / 100
    with c3:
        elast = st.slider("Trade elasticity",
                           float(CFG["policy_simulation"]["elasticity_min"]),
                           float(CFG["policy_simulation"]["elasticity_max"]),
                           float(CFG["policy_simulation"]["elasticity_base"]), 0.5)

    # Live simulation on latest month per corridor
    latest = df.sort_values(["Year","Month"]).groupby("Pair_ID").tail(1).copy()

    orig_t = latest["Transport_Cost_USD_kg"].fillna(0.062)
    orig_b = latest["Border_Friction_Cost_USD_kg"].fillna(0.045)
    new_L  = orig_t*(1-t_red) + orig_b*(1-b_red)
    orig_L = orig_t + orig_b

    gap    = latest["Price_Gap_USD_kg"].fillna(0)
    new_mis = np.where(new_L>0, (gap-new_L)/new_L, latest["MIS"].fillna(0))
    d_L    = (new_L-orig_L)/orig_L.replace(0,0.001)
    d_trade = elast*np.abs(d_L)*100

    latest["New_MIS"]         = new_mis
    latest["Delta_MIS"]       = new_mis - latest["MIS"].fillna(0)
    latest["Trade_Inc_Pct"]   = d_trade.values

    n_unlocked = int(((latest["MIS"].fillna(0)<=0) & (pd.Series(new_mis,index=latest.index)>0)).sum())
    avg_trade_inc = float(d_trade.mean())

    m1,m2,m3 = st.columns(3)
    m1.metric("Corridors Newly Viable", n_unlocked)
    m2.metric("Avg Trade Increase", f"{avg_trade_inc:.1f}%")
    m3.metric("Avg Logistics Reduction",
              f"{float((1-new_L/orig_L).mean()*100):.1f}%")

    # Before/after MIS
    st.subheader("MIS Before vs After (current period)")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current MIS",
                          x=latest["Pair_ID"].tolist(), y=latest["MIS"].tolist(),
                          marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Simulated MIS",
                          x=latest["Pair_ID"].tolist(), y=list(new_mis),
                          marker_color="#2ca02c"))
    fig.add_hline(y=0, line_color="red", line_dash="dot", annotation_text="Threshold")
    fig.update_layout(barmode="group", height=360, yaxis_title="MIS",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # Estimated trade increase
    st.subheader("Estimated Trade Increase by Corridor")
    fig2 = px.bar(latest.sort_values("Trade_Inc_Pct"),
                   x="Trade_Inc_Pct", y="Pair_ID", orientation="h",
                   labels={"Trade_Inc_Pct":"Trade increase (%)","Pair_ID":"Corridor"},
                   color="Trade_Inc_Pct", color_continuous_scale="Greens")
    fig2.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Pre-computed scenario table
    if headlines:
        st.subheader("AfCFTA Scenario Comparison (pre-computed, cross-validated)")
        rows = []
        for sname, data in headlines.items():
            rows.append({
                "Scenario":         sname,
                "Volume (tonnes)":  f"{data.get('total_inc_vol_tonnes',0):,}",
                "Value (USD)":      f"${data.get('total_inc_value_usd',0):,}",
                "Months Unlocked":  data.get("total_months_unlocked",0),
                "Corridors Activated": data.get("corridors_activated",0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption(
        f"Elasticity {elast}: a 1% reduction in logistics costs increases trade by {elast}%. "
        f"WB range for SSA staple grains: "
        f"{CFG['policy_simulation']['elasticity_min']}–"
        f"{CFG['policy_simulation']['elasticity_max']}."
    )


# ---------------------------------------------------------------------------
# PAGE 4 — EARLY WARNING PANEL
# ---------------------------------------------------------------------------
def page_early_warning(df, outputs):
    st.title("🚨 Early Warning Panel")
    st.markdown("Food security risk flags, crisis signals, pre-harvest scarcity periods.")

    latest_yr = int(df["Year"].max())
    latest_mo = int(df[df["Year"]==latest_yr]["Month"].max())
    latest    = df[(df["Year"]==latest_yr)&(df["Month"]==latest_mo)].copy()

    # Active flags
    flag_cols = ["Crisis_Signal","Food_Security_Risk_Flag"]
    active = [c for c in flag_cols if c in latest.columns]
    if active:
        crisis_mask = pd.Series(False, index=latest.index)
        for c in active:
            crisis_mask |= (latest[c]==1)
        flagged = latest[crisis_mask]
        if len(flagged)>0:
            st.error(f"⚠ {len(flagged)} corridor(s) with active risk signals — "
                     f"{latest_mo}/{latest_yr}")
            disp = [c for c in ["Pair_ID","Exporter","Importer","MIS"] + active
                    if c in flagged.columns]
            st.dataframe(flagged[disp].sort_values("MIS",ascending=False),
                          use_container_width=True)
        else:
            st.success(f"No active risk signals in {latest_mo}/{latest_yr}")

    st.divider()

    # MIS heatmap across years
    st.subheader("Annual Average MIS by Corridor")
    pivot = df.pivot_table(index="Pair_ID", columns="Year", values="MIS", aggfunc="mean")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, aspect="auto",
                     labels=dict(color="MIS"),
                     title="Red=inefficient, Green=converged")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Deficit severity
    if "Importer_Deficit_Severity" in df.columns:
        st.subheader("Highest Importer Deficit Severity Observations")
        risk = df.nlargest(15,"Importer_Deficit_Severity")[[
            "Pair_ID","Year","Month","Importer",
            "Importer_Deficit_Severity","MIS","Food_Security_Risk_Flag"
        ]]
        st.dataframe(risk, use_container_width=True)

    # Pre-harvest calendar
    if "Importer_Pre_Harvest_Scarcity" in df.columns:
        st.subheader("Pre-Harvest Scarcity Calendar")
        mnames = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        for country in COUNTRIES:
            c_data = df[df["Importer"]==country].groupby("Month")["Importer_Pre_Harvest_Scarcity"].mean()
            sc_months = [mnames[m] for m in c_data[c_data>0.5].index.tolist()]
            if sc_months:
                st.write(f"**{country}**: pre-harvest scarcity typically — {', '.join(sc_months)}")

    # Top recommended corridors
    if "corridor_rankings" in outputs:
        rnk = outputs["corridor_rankings"]
        latest_rnk = rnk[(rnk["Year"]==latest_yr)&(rnk["Month"]==latest_mo)]
        if "Recommendation" in latest_rnk.columns:
            recs = latest_rnk[latest_rnk["Recommendation"]==1]
            if len(recs)>0:
                st.subheader("Recommended Intervention Corridors")
                for _,row in recs.iterrows():
                    arb_flag = " 🔴 Active arbitrage" if row.get("Active_Arbitrage",0) else ""
                    st.write(
                        f"**{row['Pair_ID']}** — "
                        f"Score: {row.get('Trade_Score_Final',0):.3f} | "
                        f"MIS: {row.get('MIS',0):.3f}{arb_flag}"
                    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title=CFG["dashboard"]["title"],
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("AATIP")
    st.sidebar.caption("AfCFTA AgroTrade Intelligence Platform")
    st.sidebar.markdown(
        "_Continental-scale agricultural market transmission and coordination "
        "intelligence system._"
    )
    st.sidebar.divider()

    page = st.sidebar.radio("Navigate", [
        "1. Trade Opportunity Map",
        "2. Corridor Deep-Dive",
        "3. Policy Simulator",
        "4. Early Warning Panel",
    ])
    st.sidebar.divider()
    st.sidebar.caption(
        "Central thesis: markets that function efficiently see price convergence "
        "after accounting for transport and friction costs. AATIP diagnoses when "
        "and why they do not."
    )

    try:
        df, outputs, headlines = load_all()
    except FileNotFoundError as e:
        st.error(f"Data not found: {e}. Run the pipeline first.")
        st.stop()

    if "Map" in page:
        page_map(df, outputs)
    elif "Corridor" in page:
        page_corridor(df, outputs)
    elif "Policy" in page:
        page_policy(df, headlines)
    elif "Warning" in page:
        page_early_warning(df, outputs)


if __name__ == "__main__":
    main()
