# app.py — Hedge-Fund–Grade Macro Simulator with Empirical Backdrop and Asset Anchors

import streamlit as st
import pandas as pd
import numpy as np

# === Page Config ===
st.set_page_config(page_title="Hedge-Fund Macro Simulator", layout="wide")

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.02

# === Sidebar: Empirical Macro Backdrop ===
LIQ_WEIGHTS = [0.4, 0.3, 0.3]
FISCAL_WEIGHTS = [0.33, 0.33, 0.34]
GEO_WEIGHTS = [0.5, 0.3, 0.2]

def normalize_liquidity(fed_bs, short_rate, m2):
    comps = [
        (fed_bs - 15) / 30,
        np.clip((5 - short_rate) / 5, 0, 1),
        np.clip(m2 / 15, 0, 1),
    ]
    return sum(w * c for w, c in zip(LIQ_WEIGHTS, comps))

def normalize_fiscal(deficit, gov_spend, transfers):
    comps = [
        np.clip((deficit - 1) / 14, 0, 1),
        np.clip((gov_spend - 15) / 20, 0, 1),
        np.clip((transfers - 5) / 15, 0, 1),
    ]
    return sum(w * c for w, c in zip(FISCAL_WEIGHTS, comps))

def normalize_geo(geo_idx, vix, conflicts):
    comps = [
        np.clip((geo_idx - 50) / 150, 0, 1),
        np.clip((vix - 10) / 40, 0, 1),
        np.clip((conflicts - 10) / 50, 0, 1),
    ]
    return sum(w * c for w, c in zip(GEO_WEIGHTS, comps))

def macro_conditions():
    st.sidebar.header("1. Empirical Macro Backdrop")
    use_emp = st.sidebar.checkbox("Use empirical macro inputs", True)
    if use_emp:
        override = st.sidebar.checkbox("Override backdrop manually", False)
        disabled = not override
        fed_bs = st.sidebar.number_input("Fed Balance Sheet (% GDP)", 38.0, disabled=disabled)
        short_rate = st.sidebar.number_input("Real Short-Term Rate (%)", 2.5, disabled=disabled)
        m2 = st.sidebar.number_input("M2 Growth YoY (%)", 6.0, disabled=disabled)
        deficit = st.sidebar.number_input("Budget Deficit (% GDP)", 5.0, disabled=disabled)
        gov_spend = st.sidebar.number_input("Government Spending (% GDP)", 24.0, disabled=disabled)
        transfers = st.sidebar.number_input("Net Transfers (% GDP)", 12.0, disabled=disabled)
        geo_idx = st.sidebar.number_input("Geo Risk Index", 100.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX Index", 16.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict Events", 20, disabled=disabled)
        liq = normalize_liquidity(fed_bs, short_rate, m2)
        fiscal = normalize_fiscal(deficit, gov_spend, transfers)
        geo = normalize_geo(geo_idx, vix, conflicts)
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, fiscal, disabled=not override)
        geo = st.sidebar.slider("Geo Risk", 0.0, 1.0, geo, disabled=not override)
    else:
        short_rate, m2 = 2.5, 6.0
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geo Risk", 0.0, 1.0, 0.1)
    return liq, fiscal, geo, short_rate, m2

# === Calibration ===
# Upload historical regime observations to calibrate auto-probability weights
st.sidebar.header("0. Calibration (Optional)")
hist_file = st.sidebar.file_uploader("Upload historical macro+regime CSV", type="csv")
COEFFS = None
if hist_file is not None:
    hist = pd.read_csv(hist_file)
    # Expect columns: liq, fiscal, geo, regime
    X = hist[['liq','fiscal','geo']].values
    X = np.hstack([np.ones((X.shape[0],1)), X])  # intercept
    Y = pd.get_dummies(hist['regime']).reindex(columns=['Expansion','Recession','Stagflation','Deflation']).values
    # Least squares fit
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    COEFFS = B  # shape (4 regimes x 4 coefficients)

# === Sidebar: Market Inputs & Geo Flashpoints ===
def market_inputs():
    st.sidebar.header("2. Market Inputs & Geo Flashpoints")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", value=220.0)
    spx = st.sidebar.number_input("Current SPX Index", value=4500.0)
    st.sidebar.markdown("**Actual Asset Prices**")
    a_gold = st.sidebar.number_input("Current Gold Price ($)", value=1900.0)
    a_oil = st.sidebar.number_input("Current Oil Price ($)", min_value=0.0, value=75.0)
    a_10y = st.sidebar.number_input("Current 10Y Yield (%)", value=3.5)
    st.sidebar.markdown("**Geo Flashpoint Impacts**")
    geo_events = {}
    for name in ["China–US Tariff Escalation", "Russia Gas Cutoff"]:
        w = st.sidebar.slider(f"Impact: {name}", -1.0, 1.0, 0.0, key=name)
        geo_events[name] = w
    num_custom = st.sidebar.number_input("# Custom Geo Events", 0, 5, 0)
    for i in range(num_custom):
        cname = st.sidebar.text_input(f"Event {i+1} Name", key=f"cn{i}")
        if cname:
            w = st.sidebar.slider(f"Impact: {cname}", -1.0, 1.0, 0.0, key=f"cw{i}")
            geo_events[cname] = w
    return eps, spx, a_gold, a_oil, a_10y, geo_events

# === Utility Functions ===
def pe_from_real(rate_pct, prem=0.04):
    """
    *Calculates a baseline P/E anchor by inverting the required return (real rate in decimal + equity premium), with percent input.*
    """
    # Convert percentage input to decimal
    real_rate = rate_pct / 100.0
    required_return = real_rate + prem
    pe = 1 / required_return if required_return > 0 else float('inf')
    return min(40, max(8, pe))

def regimes_and_probs(liq, fiscal, geo):
    """
    *Automatically computes regime probabilities based on macro backdrop scores,*
    *then allows manual override if desired.*
    """
    st.sidebar.header("3. Regime Probabilities")
    # Compute automatic probabilities from macro inputs
    # Expansion driven by liquidity and fiscal
    exp_p = max(liq * 0.6 + fiscal * 0.4 - geo * 0.2, 0)
    # Recession driven by low liquidity and high geo risk
    rec_p = max((1 - liq) * 0.7 + geo * 0.3, 0)
    # Stagflation driven by moderate growth and medium geo-fiscal mix
    stag_p = max(fiscal * 0.2 + geo * 0.5 + liq * 0.1, 0)
    # Deflation driven by low fiscal and low geo risk
    defl_p = max((1 - fiscal) * 0.5 + (1 - geo) * 0.5 - liq * 0.2, 0)
    total = exp_p + rec_p + stag_p + defl_p
    # Normalize to sum to 1
    auto_probs = {
        'Expansion': exp_p/total,
        'Recession': rec_p/total,
        'Stagflation': stag_p/total,
        'Deflation': defl_p/total
    }
    # Manual override toggle
    override = st.sidebar.checkbox("Override probabilities manually", False)
    probs = {}
    for r in ['Expansion','Recession','Stagflation','Deflation']:
        default_pct = int(auto_probs[r] * 100)
        probs[r] = st.sidebar.slider(
            f"P({r})%", 0, 100, default_pct,
            disabled=not override,
            key=f"prob_{r}"
        )
    # If not override, use automatic
    if not override:
        probs = {r: int(auto_probs[r]*100) for r in auto_probs}
    # Ensure sum to 100 when overriding
    if override and sum(probs.values()) != 100:
        st.sidebar.error("Sum must equal 100% when manual override is used.")
        st.stop()
    return ['Expansion','Recession','Stagflation','Deflation'], probs


def nelson_siegel(rt):
    return 1 - ((1-np.exp(-rt))/rt) + 0.5*(((1-np.exp(-rt))/rt)-np.exp(-rt))

def price_gold(rt, vix, geo_score):
    return 2000*(1-rt*0.1) + vix*10 + geo_score*300

def price_oil(inv, opec, pmi, geo_score):
    """
    *Prices crude oil by incorporating inventory changes, OPEC quota shocks, PMI-driven demand, and geo shocks.*
    """
    return 80 * (1 + pmi/100 - inv/100) + opec * 80 + geo_score * 50

    return 80*(1 + pmi/100 - inv/100) + opec*80 + geo_score*50

def eps_proj(eps, gdp, inf, ratec, sharec):
    rev = eps*(1+gdp/100)
    marg = rev*(1-inf*0.005-ratec*0.01)
    debt_cost = ratec * 0.1
    floor = eps * 0.125
    return max(marg - debt_cost, floor)*(1-sharec)

def define_scenarios():
    return {
        'Expansion':   {'pe_mul':1.2},
        'Recession':   {'pe_mul':0.8},
        'Stagflation': {'pe_mul':0.9},
        'Deflation':   {'pe_mul':0.85}
    }

def macro_mult(liq, fiscal, geo):
    impact = liq*0.25 + fiscal*0.2 - geo*0.3
    return 1 + min(MAX_MACRO_PE_IMPACT, impact)

# === Portfolio Editor ===
def portfolio_editor():
    st.subheader("4. Portfolio Allocation")
    assets = ['Equities','Gold','Oil','Bonds','Cash']
    df_init = pd.DataFrame({'Asset': assets, 'Pct': [40,20,20,15,5]})
    df = st.data_editor(df_init, use_container_width=True) if hasattr(st,'data_editor') else st.experimental_data_editor(df_init, use_container_width=True)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Portfolio weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values/100

# === Monte Carlo Simulation ===
def simulate(alloc, ret, cov, sims=3000):
    draws = np.random.multivariate_normal(ret, cov, sims)
    return (draws * alloc).sum(axis=1)

# === Main Application ===
def run():
    liq, fiscal, geo, rt, m2 = macro_conditions()
    eps, spx, a_gold, a_oil, a_10y, geo_events = market_inputs()
    geo_score = sum(geo_events.values())
    regimes, probs = regimes_and_probs(liq, fiscal, geo)

    # Scenario Drivers & Correlations
    st.sidebar.header("5. Scenario Drivers & Correlations")
    gdp_defaults = {'Expansion': 3.0, 'Recession': -1.0, 'Stagflation': 1.0, 'Deflation': -0.5}
    rate_defaults = {'Expansion': 0.2, 'Recession': 1.0, 'Stagflation': 0.8, 'Deflation': -0.2}
    share_defaults = {'Expansion': 0.0, 'Recession': 0.02, 'Stagflation': 0.0, 'Deflation': 0.0}
    values, rets, eps_list, pe_list, corr_vals = [], [], [], [], {}
    for reg in regimes:
        with st.sidebar.expander(reg, True):
            gdp = st.number_input(f"GDP Growth {reg}%", gdp_defaults[reg], key=f"gdp{reg}")
            ratec = st.number_input(f"Rate Shock {reg}%", rate_defaults[reg], key=f"rs{reg}")
            sharec = st.number_input(f"Share Chg {reg}%", share_defaults[reg], key=f"sc{reg}")
            corr_vals[reg] = st.slider(f"Eq-Gold Corr {reg}", -1.0, 1.0, -0.2, key=f"c{reg}")
        eps_f = eps_proj(eps, gdp, m2, ratec, sharec)
        pe_f = pe_from_real(rt) * macro_mult(liq, fiscal, geo)
        fair_spx_reg = eps_f * pe_f
        values.append(fair_spx_reg)
        rets.append(fair_spx_reg / spx - 1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)

    weighted_eps = sum(probs[r]/100 * eps_list[i] for i,r in enumerate(regimes))
    weighted_pe  = sum(probs[r]/100 * pe_list[i]  for i,r in enumerate(regimes))
    fair_spx = weighted_eps * weighted_pe

    # Display regime valuation table
    dfv = pd.DataFrame({'Regime': regimes, 'Fair SPX': values, 'Return%': rets, 'P%': [probs[r] for r in regimes]})
    st.write(dfv)

        # 6. Valuation Anchors Comparison
    st.subheader("6. Valuation & Asset Price Anchors")
    gold_price = price_gold(
        rt,
        st.sidebar.number_input("VIX for Gold", value=16.0),
        geo_score
    )
    oil_price = price_oil(
        st.sidebar.number_input("Oil Inv Change (%)", value=0.0),
        st.sidebar.slider("OPEC Quota", -1.0, 1.0, 0.0),
        st.sidebar.number_input("Global PMI", value=50.0),
        geo_score
    )
    bond_yield = nelson_siegel(rt)

    df_anchors = pd.DataFrame({
        'Metric': ["Current SPX","Weighted EPS","Weighted P/E","Gold","Oil","10Y Yield"],
        'Actual': [spx, eps, spx/eps, a_gold, a_oil, a_10y/100],
        'Model':  [fair_spx, weighted_eps, weighted_pe, gold_price, oil_price, bond_yield]
    }).set_index('Metric')
    st.table(df_anchors.style.format({"Actual":"{:.2f}","Model":"{:.2f}"}))

    # 7. Portfolio & Expected Return
    dfp, alloc = portfolio_editor()
    exp_eq = sum(probs[r]/100 * rets[i] for i,r in enumerate(regimes))
    ret_asset = np.array([exp_eq, gold_price/a_gold-1, oil_price/a_oil-1, bond_yield, DEFAULT_CASH_YIELD])
    exp_return = alloc @ ret_asset
    st.subheader("7. Expected Portfolio Return")
    st.metric("Expected Return", f"{exp_return:.2%}")

    # 8. Monte Carlo Simulation & Distribution
    vols = np.array([0.15,0.10,0.12,0.08,0.00])
    avg_corr = np.mean(list(corr_vals.values()))
    cov = np.diag(vols) @ np.array([
        [1, avg_corr,0,0,0],
        [avg_corr,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1]
    ]) @ np.diag(vols)
    sims = simulate(alloc, ret_asset, cov)
    st.subheader("8. Portfolio MC Distribution")
    st.line_chart(pd.Series(sims).rolling(50).mean())

if __name__ == '__main__':
    run()
