# app.py — Enhanced Streamlit Macro Portfolio Simulator
# Implements advanced macro-to-asset modeling: yield curve, regime switching, Monte Carlo, inter-asset correlations

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal

# === Page Configuration ===
st.set_page_config(
    page_title="Advanced Macro Portfolio Simulator",
    layout="wide",
)

# === Constants & Defaults ===
MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.03
BOND_DURATION = 7
BASE_SPX_PE = 16

# === Sidebar: User Inputs ===
def sidebar_inputs():
    st.sidebar.header("1. Market & Macro Inputs")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", value=200.0)
    spx = st.sidebar.number_input("Current SPX Index", value=5300.0)

    # Real yields via TIPS breakeven
    inflation_expectation = st.sidebar.number_input(
        "Inflation Expectation (TIPS breakeven %)", value=2.5
    )
    nominal_rate = st.sidebar.number_input(
        "10Y Nominal Yield (%)", value=4.0
    )
    real_short_rate = nominal_rate - inflation_expectation

    # Oil supply shock input
    oil_supply_shock = st.sidebar.slider(
        "Oil Supply Shock Factor", min_value=0.0, max_value=1.0, value=0.2,
        help="0 = no shock, 1 = extreme supply disruption"
    )

    # Scenario probabilities
    st.sidebar.header("2. Scenario Probabilities")
    scen_names = ["Boom", "Recession", "Stagflation", "Deflation"]
    probs = {s: st.sidebar.slider(f"{s} (%)", 0, 100, 25) for s in scen_names}
    total = sum(probs.values())
    st.sidebar.markdown(f"**Total: {total}%**")
    if total != 100:
        st.sidebar.error("Probabilities must sum to 100%.")
        st.stop()

    # Correlation adjustments
    st.sidebar.header("3. Inter-Asset Correlations")
    corr_eq_gold = st.sidebar.slider("Equities <-> Gold", -1.0, 1.0, value=-0.2)
    corr_eq_bonds = st.sidebar.slider("Equities <-> Bonds", -1.0, 1.0, value=0.3)
    corr_gold_bonds = st.sidebar.slider("Gold <-> Bonds", -1.0, 1.0, value=-0.1)
    corr_matrix = np.array([
        [1.0, corr_eq_gold, corr_eq_bonds],
        [corr_eq_gold, 1.0, corr_gold_bonds],
        [corr_eq_bonds, corr_gold_bonds, 1.0],
    ])

    return {
        "eps": eps,
        "spx": spx,
        "real_short_rate": real_short_rate,
        "inflation": inflation_expectation,
        "oil_shock": oil_supply_shock,
        "probs": probs,
        "corr": corr_matrix,
    }

# === Macro Models ===
def yield_curve_factor(real_rate):
    # Nelson-Siegel placeholder: simple function mapping real rate to PE anchor
    # deeper models can replace this
    return min(40, max(8, 1/(real_rate + 0.04)))


def asset_price_targets(real_rate, inflation, oil_shock):
    # Gold: negative real yields + risk premium
    gold_price = 2000 * (1 + max(0, 0.1 - real_rate)*0.5)

    # Oil: base price + supply shock + inflation
    crude_price = 80 * (1 + oil_shock*0.3 + inflation*0.02)

    # Bond yields derived
    bond_yield = (real_rate + inflation) * (1 + oil_shock*0.1)
    return {
        "Gold": gold_price,
        "Crude": crude_price,
        "10Y": bond_yield,
    }

# === Scenario Definitions ===
def define_scenarios(real_rate):
    # scenarios use dynamic PE and eps drivers
    base_pe = yield_curve_factor(real_rate)
    return {
        "Boom":        {"pe": base_pe*1.1, "eps_delta": +0.15},
        "Recession":   {"pe": base_pe*0.8, "eps_delta": -0.20},
        "Stagflation": {"pe": base_pe*0.9, "eps_delta": -0.10},
        "Deflation":   {"pe": base_pe*0.85,"eps_delta": -0.05},
    }

# === EPS Driver ===
def eps_driver(eps, eps_delta, gdp_growth=2.0, margin_squeeze=0.01):
    # revenue effect + margin effect
    return eps * (1 + eps_delta + gdp_growth*0.01 - margin_squeeze)

# === Monte Carlo Simulation ===
def simulate_runs(inputs, scenarios, targets, corr, n_sims=1000):
    # assets: [Equity, Gold, Bonds]
    mu = np.array([
        # expected returns under current estimate
        (targets['Gold']/2000 -1),
        (targets['Crude']/80  -1),
        (targets['10Y']/4 -1),
    ])
    # covariance: assume vol of 10% each
    sigma = np.diag([0.10, 0.15, 0.05])
    cov = sigma @ corr @ sigma
    sims = multivariate_normal.rvs(mean=mu, cov=cov, size=n_sims)
    return sims

# === Main App ===
def run():
    inputs = sidebar_inputs()
    eps = inputs['eps']; spx = inputs['spx']
    real_rate = inputs['real_short_rate']
    inflation = inputs['inflation']
    oil_shock = inputs['oil_shock']
    probs = inputs['probs']; corr = inputs['corr']

    # Build scenarios dynamically
    scenarios = define_scenarios(real_rate)

    # Calculate weighted EPS & PE
    weighted_eps = sum(
        probs[s]/100 * eps_driver(eps, scenarios[s]['eps_delta'])
        for s in scenarios
    )
    weighted_pe = sum(
        probs[s]/100 * scenarios[s]['pe']
        for s in scenarios
    )
    macro_mult = 1 + min(MAX_MACRO_PE_IMPACT, (weighted_pe/base_pe -1))
    fair_spx = weighted_eps * weighted_pe * macro_mult

    # Asset price targets
    targets = asset_price_targets(real_rate, inflation, oil_shock)

    # Display summary
    st.title("Advanced Macro Portfolio Simulator")
    st.subheader("Valuation Summary")
    trailing_pe = spx/eps
    st.markdown(f"**Trailing P/E:** {trailing_pe:.1f}")
    st.markdown(f"**Weighted EPS:** {weighted_eps:.1f}")
    st.markdown(f"**Weighted PE:** {weighted_pe:.1f}")
    st.markdown(f"**Fair SPX:** {fair_spx:,.0f}")

    # Asset anchors
    st.subheader("Asset Price Anchors")
    st.write(pd.DataFrame.from_dict(targets, orient='index', columns=['Value']))

    # Portfolio Editor
    df = pd.DataFrame({
        'Asset': ['Equities', 'Gold', 'Bonds', 'Cash'],
        'AllocationPct': [50, 20, 20, 10]
    })
    df = st.experimental_data_editor(df, use_container_width=True)
    total_pct = df['AllocationPct'].sum()
    if total_pct != 100:
        st.error("Total allocation must equal 100%.")
    df['Allocation'] = df['AllocationPct']/100 * inputs['spx']

    # Simulations
    sims = simulate_runs(inputs, scenarios, targets, corr)
    eq_sim, gold_sim, bond_sim = sims.T
    port_sim = (
        df.loc[df['Asset']=='Equities','AllocationPct'].values/100 * eq_sim +
        df.loc[df['Asset']=='Gold','AllocationPct'].values/100 * gold_sim +
        df.loc[df['Asset']=='Bonds','AllocationPct'].values/100 * bond_sim +
        df.loc[df['Asset']=='Cash','AllocationPct'].values/100 * DEFAULT_CASH_YIELD
    )

    st.subheader("Monte Carlo Portfolio Return Distribution")
    st.write("Histogram of simulated portfolio returns:")
    hist = np.histogram(port_sim, bins=50)
    st.bar_chart(hist[0])

if __name__ == '__main__':
    run()
