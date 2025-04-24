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
MAX_MACRO_PE_IMACT = 0.6
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
        defaults = {r: int(auto[r] * 100) for r in auto}
        diff = 100 - sum(defaults.values())
        if diff != 0:
            key_max = max(defaults, key=defaults.get)
            defaults[key_max] += diff
        probs = defaults
    return list(auto.keys()), probs

# === Portfolio Allocation ===
def portfolio_editor():
    st.subheader("Portfolio Allocation")
    # Side-by-side: allocations and equity beta
    col1, col2 = st.columns([3, 1])
    with col1:
        df_init = pd.DataFrame({'Asset': ['Equities', 'Gold', 'Oil', 'Bonds', 'Cash'], 'Pct': [40, 20, 20, 15, 5]})
        if hasattr(st, 'data_editor'):
            df = st.data_editor(df_init, use_container_width=True)
        else:
            df = st.experimental_data_editor(df_init, use_container_width=True)
    with col2:
        equity_beta = st.number_input("Equity Beta", min_value=0.0, value=1.0, step=0.1)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values / 100, equity_beta

# === Scenario Drivers & Correlations ===
def step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs):
    st.sidebar.header("Scenario Drivers & Correlations")
    ... # unchanged

# === Anchor Drivers & Assumptions ===
def step6_anchors_inputs():
    ... # unchanged

# === Helper Functions ===
def pe_from_real(rate_pct, prem=0.04): ...
def nelson_siegel(rt): ...
def price_gold(rt, vix, geo_score, eq_gold_corr): ...
def price_oil(inv, opec, pmi, geo_score): ...
def eps_proj(eps, gdp, inf, ratec, sharec): ...
def macro_mult(liq, fiscal, geo): ...
def simulate(alloc, ret, cov, sims=3000): ...

# === Main Application ===
def run():
    eps, spx, a_gold, a_oil, a_10y, geo_events = step1_market()
    liq, fiscal, geo, rt, m2 = step2_backdrop()
    global regimes
    regimes, probs = step3_regimes(liq, fiscal, geo)
    values, rets, eps_list, pe_list, corr_vals = step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs)
    weighted_eps = sum(probs[r] / 100 * eps_list[i] for i, r in enumerate(regimes))
    weighted_pe = sum(probs[r] / 100 * pe_list[i] for i, r in enumerate(regimes))
    fair_spx = weighted_eps * weighted_pe
    # ... table and anchors unchanged ...
    # Portfolio Allocation
    dfp, alloc, equity_beta = portfolio_editor()
    exp_eq = sum(probs[r] / 100 * rets[i] for i, r in enumerate(regimes)) * equity_beta
    ret_asset = np.array([exp_eq, gold_price / a_gold - 1, oil_price / a_oil - 1, bond_yield, DEFAULT_CASH_YIELD])
    exp_return = alloc @ ret_asset
    st.subheader("Expected Portfolio Return")
    st.metric("Expected Return", f"{exp_return:.2%}")
    # ... rest unchanged ...

if __name__ == '__main__':
    run()
