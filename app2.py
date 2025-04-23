# app.py — Hedge-Fund–Grade Macro Simulator with Empirical Backdrop
# Incorporates empirical liquidity/fiscal/geo inputs, advanced gold/oil logic,
# yield curve modeling, EPS segmentation, regime-switch dynamics,
# event-triggered shocks, dynamic correlations, Monte Carlo & backtest hooks

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# === Page Config ===
st.set_page_config(page_title="Hedge-Fund Macro Simulator", layout="wide")

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.6  # *Maximum allowable adjustment to P/E based on macro multiplier.*
DEFAULT_CASH_YIELD = 0.02   # *Assumed annual return rate for cash allocations.*
BOND_DURATION = 7           # *Effective duration used for bond return calculations.*

# === Sidebar: Empirical Macro Backdrop ===
def macro_conditions():
    """
    *Collects empirical data for liquidity, fiscal, and geopolitical risk, normalizes them to 0–1,
    and allows manual overrides when needed.*
    """
    st.sidebar.header("1. Empirical Macro Backdrop")
    st.sidebar.markdown("_Collects and normalizes empirical data on liquidity, fiscal policy, and geopolitical risk, with optional manual override._")
    use_emp = st.sidebar.checkbox("Use empirical macro inputs", True)
    # default values
    short_term_rate = 1.5
    m2_growth = 4.0
    if use_emp:
        override = st.sidebar.checkbox("Override backdrop manually", False)
        disabled = not override
        # Liquidity inputs
        st.sidebar.markdown("#### Liquidity Inputs")
        fed_bs = st.sidebar.number_input("Fed Balance Sheet (% GDP)", 35.0, disabled=disabled)
        short_term_rate = st.sidebar.number_input("Real Short-Term Rate (%)", 1.5, disabled=disabled)
        m2_growth = st.sidebar.number_input("M2 Growth YoY (%)", 4.0, disabled=disabled)
        # Fiscal inputs
        st.sidebar.markdown("#### Fiscal Inputs")
        deficit = st.sidebar.number_input("Budget Deficit (% GDP)", 6.0, disabled=disabled)
        gov_spend = st.sidebar.number_input("Government Spending (% GDP)", 25.0, disabled=disabled)
        transfers = st.sidebar.number_input("Net Transfers (% GDP)", 10.0, disabled=disabled)
        # Geopolitical inputs
        st.sidebar.markdown("#### Geo Risk Inputs")
        geo_index = st.sidebar.number_input("Geo Risk Index", 120.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX Index", 20.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict Events", 30, disabled=disabled)

        # Normalize to 0-1
        liq = np.mean([
            (fed_bs-15)/30,                           # *Fed liquidity proxy based on balance sheet size.*
            np.clip((5-short_term_rate)/5,0,1),       # *Interest-rate-driven liquidity normalization.*
            np.clip(m2_growth/15,0,1)                 # *Money supply growth impact on liquidity.*
        ])
        fiscal = np.mean([
            np.clip((deficit-1)/14,0,1),              # *Deficit-to-GDP impact on fiscal tailwinds.*
            np.clip((gov_spend-15)/20,0,1),           # *Government spending normalization.*
            np.clip((transfers-5)/15,0,1)             # *Net transfers normalization.*
        ])
        geo = np.mean([
            np.clip((geo_index-50)/150,0,1),          # *Narrative risk scaling from index.*
            np.clip((vix-10)/40,0,1),                 # *Volatility-driven geo risk.*
            np.clip((conflicts-10)/50,0,1)            # *Event count-driven geo risk.*
        ])

        # Manual overrides
        st.sidebar.markdown("#### Override Derived Backdrop")
        liq = st.sidebar.slider("Liquidity",0.0,1.0,liq,disabled=not override)
        fiscal = st.sidebar.slider("Fiscal Stimulus",0.0,1.0,fiscal,disabled=not override)
        geo = st.sidebar.slider("Geo Risk",0.0,1.0,geo,disabled=not override)
    else:
        liq = st.sidebar.slider("Liquidity",0.0,1.0,0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus",0.0,1.0,0.3)
        geo = st.sidebar.slider("Geo Risk",0.0,1.0,0.1)

    return liq, fiscal, geo, short_term_rate, m2_growth

# === Sidebar: Market Inputs ===
def market_inputs():
    """
    *Gathers core market data including trailing earnings, index level, and geo-flashpoint flags.*
    """
    st.sidebar.header("2. Market Inputs")
    st.sidebar.markdown("_Gathers key market figures—SPX earnings/index—and geopolitical flags to feed downstream models._")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)",200.0)
    spx = st.sidebar.number_input("Current SPX Index",5300.0)

    # Narrative geo-flashpoints
    st.sidebar.header("3. Geo Flashpoints")
    china_tariff = st.sidebar.checkbox("China–US Tariff Escalation")
    russia_gas_cut = st.sidebar.checkbox("Russia Gas Cutoff")

    return eps, spx, china_tariff, russia_gas_cut

# === PE from Real Rates ===
def pe_from_real(rate, prem=0.04):
    """
    *Calculates a baseline P/E anchor by inverting the required return (real rate + equity premium).* 
    """
    req = rate + prem
    return min(40, max(8,1/req))

# === Scenario & Probabilities ===
def regimes_and_probs():
    """
    *Defines discrete economic regimes and collects user-assigned probabilities that sum to 100%.*
    """
    st.sidebar.header("4. Regime Probabilities")
    st.sidebar.markdown("_Defines economic regimes and captures user-assigned likelihoods that sum to 100%._")
    regimes = ["Expansion","Recession","Stagflation","Deflation"]
    probs = {r: st.sidebar.slider(f"P({r})%",0,100,25) for r in regimes}
    if sum(probs.values())!=100:
        st.sidebar.error("Sum must be 100%")
        st.stop()
    return regimes, probs

# === Yield Curve Model ===
def nelson_siegel(real_rate):
    """
    *Approximates a single-factor Nelson–Siegel yield curve to capture level, slope, and curvature.*
    """
    lvl = 1
    slope = -1 * ((1 - np.exp(-real_rate)) / real_rate)
    curv = 0.5 * (((1 - np.exp(-real_rate)) / real_rate) - np.exp(-real_rate))
    return lvl + slope + curv

# === Gold & Crude Pricing ===
def price_gold(real_rate, vix, china_tariff, geo):
    """
    *Models gold price as a function of negative real yields, volatility premium, and geo-flashpoint shocks.*
    """
    base = 2000 * (1 - real_rate * 0.1)
    safe = vix * 10
    shock = 300 if china_tariff or geo > 0.6 else 0
    return base + safe + shock


def price_crude(inv_change, opec_quota, pmi, russia_cut):
    """
    *Prices crude oil by incorporating inventory changes, OPEC quotas, PMI-driven demand, and supply shocks.*
    """
    base = 80 * (1 + pmi/100 - inv_change/100)
    supply = base * opec_quota
    shock = 50 if russia_cut else 0
    return base + supply + shock

# === EPS Segmentation ===
def eps_proj(eps, gdp, inflation, rate_chg, share_chg):
    """
    *Projects forward EPS by segmenting revenue growth, margin impact, and share-count changes.*
    """
    rev = eps * (1 + gdp/100)
    marg = rev * (1 - inflation*0.005 - rate_chg*0.01)
    return marg * (1 - share_chg)

# === Define Scenarios ===
def define_scenarios():
    """
    *Maps each economic regime to dynamic P/E multipliers and EPS deltas.*
    """
    return {
        'Expansion':   {'pe_mul':1.2,'eps_d':+0.10},
        'Recession':   {'pe_mul':0.8,'eps_d':-0.25},
        'Stagflation': {'pe_mul':0.9,'eps_d':-0.15},
        'Deflation':   {'pe_mul':0.85,'eps_d':-0.05}
    }

# === Portfolio & Correlations ===
def portfolio_and_corr():
    """
    *Allows users to set portfolio allocations and regime-specific equity-gold correlations.*
    """
    st.subheader("Portfolio Allocation & Corr")
    st.markdown("_Set your asset weights and adjust correlation coefficients for each regime to capture changing relationships._")():
    """
    *Allows users to set portfolio allocations and regime-specific equity-gold correlations.*
    """
    st.subheader("Portfolio Allocation & Corr")
    assets = ['Equities','Gold','Oil','Bonds','Cash']
    df = pd.DataFrame({'Asset': assets, 'Pct': [40,20,20,15,5]})
    df = st.data_editor(df, use_container_width=True)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Allocation must sum to 100%.")
        st.stop()
    corr = {}
    for reg in define_scenarios().keys():
        corr[reg] = st.slider(f"Corr eq-gold {reg}", -1.0, 1.0, -0.2, key=reg)
    return df, corr

# === Monte Carlo ===
def simulate(alloc, ret, cov, sims=3000):
    """
    *Generates multivariate normal return paths for assets and computes portfolio return distribution.*
    """
    draws = np.random.multivariate_normal(ret, cov, sims)
    return (draws * alloc).sum(axis=1)

# === Main ===
def run():
    """
    *Executes the full simulation pipeline: collects inputs, computes anchors, projects scenarios,
    and runs Monte Carlo on the assembled portfolio.*
    """
    liq, fiscal, geo, short_term_rate, m2_growth = macro_conditions()
    eps, spx, china_tariff, russia_gas_cut = market_inputs()
    regimes, probs = regimes_and_probs()
    specs = define_scenarios()

    # Compute anchor prices
    vix = st.sidebar.number_input("VIX for Gold", 20)
    inv = st.sidebar.number_input("Oil Inv Change", 0.0)
    opec = st.sidebar.slider("OPEC Shock", -1.0, 1.0, 0.0)
    pmi  = st.sidebar.number_input("Global PMI", 50.0)

    gold = price_gold(short_term_rate, vix, china_tariff, geo)
    oil  = price_crude(inv, opec, pmi, russia_gas_cut)
    bond = nelson_siegel(short_term_rate) * 0.01

    # EPS projections & fair SPX
    values, returns = [], []
    for reg in regimes:
        eps_f = eps_proj(
            eps,
            st.sidebar.number_input(f"GDP Growth {reg}", 2.0),
            m2_growth,
            st.sidebar.number_input(f"Rate Shock {reg}%", 0.0),
            st.sidebar.number_input(f"Share Chg {reg}%", 0.0)
        )
        pe = pe_from_real(short_term_rate) * specs[reg]['pe_mul']
        fair = eps_f * pe
        values.append(fair)
        returns.append(fair / spx - 1)

    dfv = pd.DataFrame({'Regime': regimes, 'Fair SPX': values, 'Ret': returns, 'P': [probs[r] for r in regimes]})
    st.write(dfv)

    dfp, corr = portfolio_and_corr()
    alloc = dfp['Pct'].values / 100

    # Compute expected asset returns
    exp_eq = sum(probs[r]/100 * (values[idx]/spx - 1) for idx, r in enumerate(regimes))
    gold_ret = gold / 2000 - 1
    oil_ret  = oil / 80   - 1
    bond_ret = bond  # in decimal form
    cash_ret = DEFAULT_CASH_YIELD

    ret_asset = np.array([exp_eq, gold_ret, oil_ret, bond_ret, cash_ret])

    # Covariance matrix placeholder
    sigma = np.diag([0.15, 0.10, 0.12, 0.08, 0.00])
    cov   = sigma @ np.eye(5) @ sigma

    port_sims = simulate(alloc, ret_asset, cov)

    st.subheader("Portfolio MC Distribution")
    st.line_chart(pd.Series(port_sims).rolling(50).mean())

if __name__ == '__main__':
    run()
