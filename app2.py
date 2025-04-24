# app.py — Macro Scenario Simulator: easy, guided, and transparent

import streamlit as st
import pandas as pd
import numpy as np

# === Page Config & Title ===
st.set_page_config(page_title="Macro Scenario Simulator", layout="wide")
st.title("Macro Scenario Simulator")
st.markdown(
    "This interactive tool helps you stress-test a portfolio across different economic regimes. "
    "Use the sidebar titles to enter market prices, set the macro backdrop, "
    "adjust scenario drivers, and visualize fair-value estimates and risk distributions."
)

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.02

# === Market Prices & Flashpoints ===
def step1_market():
    st.sidebar.header("Market Prices & Flashpoints")
    st.sidebar.markdown("*Enter asset prices and geo-event impacts.*")
    eps = st.sidebar.number_input("SPX trailing EPS", value=220.0, min_value=0.0)
    spx = st.sidebar.number_input("SPX index level", value=4500.0, min_value=0.0)
    st.sidebar.markdown("**Current Asset Prices**")
    a_gold = st.sidebar.number_input("Gold price ($)", value=1900.0, min_value=0.0)
    a_oil = st.sidebar.number_input("Oil price ($)", value=75.0, min_value=0.0)
    a_10y = st.sidebar.number_input("10Y yield (%)", value=3.5, min_value=0.0)
    st.sidebar.markdown("**Geo Flashpoints**")
    geo_events = {}
    for name in ["Tariff shock", "Gas cutoff"]:
        geo_events[name] = st.sidebar.slider(f"Impact: {name}", -1.0, 1.0, 0.0)
    return eps, spx, a_gold, a_oil, a_10y, geo_events

# === Macro Backdrop ===
LIQ_WEIGHTS = [0.4, 0.3, 0.3]
FISCAL_WEIGHTS = [0.33, 0.33, 0.34]
GEO_WEIGHTS = [0.5, 0.3, 0.2]

def normalize_liquidity(fed_bs, short_rate, m2):
    comps = [
        (fed_bs - 15) / 30,
        np.clip((5 - short_rate) / 5, 0, 1),
        np.clip(m2 / 15, 0, 1)
    ]
    return sum(w * c for w, c in zip(LIQ_WEIGHTS, comps))

def normalize_fiscal(deficit, spend, transfer):
    comps = [
        np.clip((deficit - 1) / 14, 0, 1),
        np.clip((spend - 15) / 20, 0, 1),
        np.clip((transfer - 5) / 15, 0, 1)
    ]
    return sum(w * c for w, c in zip(FISCAL_WEIGHTS, comps))

def normalize_geo(idx, vix, conflicts):
    comps = [
        np.clip((idx - 50) / 150, 0, 1),
        np.clip((vix - 10) / 40, 0, 1),
        np.clip((conflicts - 10) / 50, 0, 1)
    ]
    return sum(w * c for w, c in zip(GEO_WEIGHTS, comps))

def step2_backdrop():
    st.sidebar.header("Macro Backdrop")
    st.sidebar.markdown("*Empirical inputs or overrides for liquidity, fiscal, and geo risk.*")
    use_emp = st.sidebar.checkbox("Use empirical inputs", True)
    if use_emp:
        override = st.sidebar.checkbox("Manual override", False)
        disabled = not override
        fed_bs = st.sidebar.number_input("Fed BS (% of GDP)", value=38.0, disabled=disabled)
        short_rate = st.sidebar.number_input("Real short rate (%)", value=2.5, disabled=disabled)
        m2 = st.sidebar.number_input("M2 growth YoY (%)", value=6.0, disabled=disabled)
        deficit = st.sidebar.number_input("Budget deficit (% of GDP)", value=5.0, disabled=disabled)
        spend = st.sidebar.number_input("Govt spending (% of GDP)", value=24.0, disabled=disabled)
        transfer = st.sidebar.number_input("Net transfers (% of GDP)", value=12.0, disabled=disabled)
        geo_idx = st.sidebar.number_input("Geo risk index", value=100.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX index", value=16.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict events (#)", value=20, disabled=disabled)
        liq = normalize_liquidity(fed_bs, short_rate, m2)
        fiscal = normalize_fiscal(deficit, spend, transfer)
        geo = normalize_geo(geo_idx, vix, conflicts)
        liq = st.sidebar.slider("Liquidity score", 0.0, 1.0, liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal score", 0.0, 1.0, fiscal, disabled=not override)
        geo = st.sidebar.slider("Geo risk score", 0.0, 1.0, geo, disabled=not override)
    else:
        short_rate, m2 = 2.5, 6.0
        liq = st.sidebar.slider("Liquidity score", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal score", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geo risk score", 0.0, 1.0, 0.1)
    return liq, fiscal, geo, short_rate, m2

# === Regime Probabilities ===
def step3_regimes(liq, fiscal, geo):
    st.sidebar.header("Regime Probabilities")
    st.sidebar.markdown("*Auto-computed from backdrop, optional override.*")
    exp_p = max(liq * 0.6 + fiscal * 0.4 - geo * 0.2, 0)
    rec_p = max((1 - liq) * 0.7 + geo * 0.3, 0)
    stag_p = max(fiscal * 0.2 + geo * 0.5 + liq * 0.1, 0)
    defl_p = max((1 - fiscal) * 0.5 + (1 - geo) * 0.5 - liq * 0.2, 0)
    total_score = exp_p + rec_p + stag_p + defl_p
    auto = {'Expansion': exp_p / total_score, 'Recession': rec_p / total_score,
            'Stagflation': stag_p / total_score, 'Deflation': defl_p / total_score}
    override = st.sidebar.checkbox("Override probabilities", False)
    probs = {}
    for r, pct in auto.items():
        default = int(pct * 100)
        probs[r] = st.sidebar.number_input(
            f"P({r})%", min_value=0, max_value=100, value=default,
            disabled=not override, key=f"prob_{r}"
        )
    total = sum(probs.values())
    st.sidebar.markdown(f"**Total Probability: {total}%**")
    if override and total != 100:
        st.sidebar.error("Probabilities must sum to 100% when overriding.")
        st.stop()
    if not override:
        defaults = {r: int(auto[r]*100) for r in auto}
        diff = 100 - sum(defaults.values())
        if diff != 0:
            key_max = max(defaults, key=defaults.get)
            defaults[key_max] += diff
        probs = defaults
    return list(auto.keys()), probs

# === Portfolio Allocation ===
def portfolio_editor():
    st.subheader("Portfolio Allocation")
    # Equity beta input
    equity_beta = st.sidebar.slider("Equity Beta", 0.0, 2.0, 1.0, step=0.1)
    df_init = pd.DataFrame({'Asset': ['Equities', 'Gold', 'Oil', 'Bonds', 'Cash'], 'Pct': [40, 20, 20, 15, 5]})
    if hasattr(st, 'data_editor'):
        df = st.data_editor(df_init, use_container_width=True)
    else:
        df = st.experimental_data_editor(df_init, use_container_width=True)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values / 100, equity_beta

# === Scenario Drivers & Correlations ===
def step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs):
    st.sidebar.header("Scenario Drivers & Correlations")
    gdp_def = {'Expansion': 3.0, 'Recession': -1.0, 'Stagflation': 1.0, 'Deflation': -0.5}
    rate_def = {'Expansion': 0.2, 'Recession': 1.0, 'Stagflation': 0.8, 'Deflation': -0.2}
    share_def = {'Expansion': 0.0, 'Recession': 0.02, 'Stagflation': 0.0, 'Deflation': 0.0}
    values, rets, eps_list, pe_list, corr_vals = [], [], [], [], {}
    for reg in regimes:
        with st.sidebar.expander(reg, True):
            gdp = st.number_input(f"GDP {reg}%", gdp_def[reg], key=f"gdp_{reg}")
            ratec = st.number_input(f"Rate shock {reg}%", rate_def[reg], key=f"rs_{reg}")
            sharec = st.number_input(f"Share change {reg}%", share_def[reg], key=f"sc_{reg}")
            corr_vals[reg] = st.sidebar.slider(f"Eq-Gold correlation {reg}", -1.0, 1.0, -0.2, key=f"corr_{reg}")
        # compute projected EPS, P/E, fair SPX, and return
        eps_f = eps_proj(eps, gdp, m2, ratec, sharec)
        pe_f = pe_from_real(rt) * macro_mult(liq, fiscal, geo)
        fair_reg = eps_f * pe_f
        values.append(fair_reg)
        rets.append(fair_reg / spx - 1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)
    return values, rets, eps_list, pe_list, corr_vals

# === Anchor Drivers & Assumptions ===
def step6_anchors_inputs():
    st.sidebar.header("Anchor Drivers & Assumptions")
    vix_model = st.sidebar.number_input("VIX for Gold", value=16.0)
    inv_change = st.sidebar.number_input("Oil inventory change (%)", value=0.0)
    opec_quota = st.sidebar.slider("OPEC quota adjustment", -1.0, 1.0, 0.0)
    pmi_model = st.sidebar.number_input("Global PMI", value=50.0)
    return vix_model, inv_change, opec_quota, pmi_model

# === Helper Functions and Simulation same as before... ===
# (omitted for brevity)
