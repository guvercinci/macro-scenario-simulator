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
        fed_bs = st.sidebar.number_input("Fed Balance Sheet (% GDP)", 38.0, disabled=disabled)  # Default ~38%, disabled=disabled)
        short_rate = st.sidebar.number_input("Real Short-Term Rate (%)", 2.5, disabled=disabled)  # Default ~2.5%
        m2 = st.sidebar.number_input("M2 Growth YoY (%)", 6.0, disabled=disabled)  # Default ~6%
        deficit = st.sidebar.number_input("Budget Deficit (% GDP)", 5.0, disabled=disabled)  # Default ~5%
        gov_spend = st.sidebar.number_input("Government Spending (% GDP)", 24.0, disabled=disabled)  # Default ~24%
        transfers = st.sidebar.number_input("Net Transfers (% GDP)", 12.0, disabled=disabled)  # Default ~12%
        geo_idx = st.sidebar.number_input("Geo Risk Index", 100.0, disabled=disabled)  # Default moderate risk
        vix = st.sidebar.number_input("VIX Index", 16.0, disabled=disabled)  # Default ~16%
        conflicts = st.sidebar.number_input("Conflict Events", 20, disabled=disabled)  # Default ~20 events
        liq = normalize_liquidity(fed_bs, short_rate, m2)
        fiscal = normalize_fiscal(deficit, gov_spend, transfers)
        geo = normalize_geo(geo_idx, vix, conflicts)
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, fiscal, disabled=not override)
        geo = st.sidebar.slider("Geo Risk", 0.0, 1.0, geo, disabled=not override)
    else:
        short_rate, m2 = 1.5, 4.0
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geo Risk", 0.0, 1.0, 0.1)
    return liq, fiscal, geo, short_rate, m2

# === Sidebar: Market Inputs & Flashpoints ===
def market_inputs():
    st.sidebar.header("2. Market Inputs & Geo Flashpoints")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", 220.0)  # Default ~$220
    spx = st.sidebar.number_input("Current SPX Index", 4500.0)  # Default ~$4500
    st.sidebar.markdown("**Actual Asset Prices**")
    a_gold = st.sidebar.number_input("Current Gold Price ($)", 1900.0)  # Default ~$1900
    a_oil = st.sidebar.number_input("Current Oil Price ($)", 75.0)  # Default ~$75
    a_10y = st.sidebar.number_input("Current 10Y Yield (%)", 3.5)  # Default ~3.5%
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
def pe_from_real(rate, prem=0.04):
    return min(40, max(8, 1/(rate+prem)))

def regimes_and_probs():
    st.sidebar.header("3. Regime Probabilities")
    regimes = ["Expansion", "Recession", "Stagflation", "Deflation"]
    probs = {r: st.sidebar.slider(f"P({r})%", 0, 100, 25) for r in regimes}
    if sum(probs.values()) != 100:
        st.sidebar.error("Regime probabilities must sum to 100%.")
        st.stop()
    return regimes, probs

def nelson_siegel(rt):
    return 1 - ((1-np.exp(-rt))/rt) + 0.5*(((1-np.exp(-rt))/rt)-np.exp(-rt))

def price_gold(rt, vix, geo_score):
    return 2000*(1-rt*0.1) + vix*10 + geo_score*300

def price_oil(inv, opec, pmi, geo_score):
    return 80*(1 + pmi/100 - inv/100) + opec*80 + geo_score*50

def eps_proj(eps, gdp, inf, ratec, sharec):
    rev = eps*(1+gdp/100)
    marg = rev*(1-inf*0.005-ratec*0.01)
    # debt service cost
    debt_cost = ratec * 0.1
    floor = eps * 0.125
    return max(marg - debt_cost, floor)*(1-sharec)

def define_scenarios():
    return {
        'Expansion':   {'pe_mul':1.2,'eps_d':+0.10},
        'Recession':   {'pe_mul':0.8,'eps_d':-0.25},
        'Stagflation': {'pe_mul':0.9,'eps_d':-0.15},
        'Deflation':   {'pe_mul':0.85,'eps_d':-0.05}
    }
def macro_mult(liq, fiscal, geo):
    impact = liq*0.25 + fiscal*0.2 - geo*0.3
    return 1 + min(MAX_MACRO_PE_IMPACT, impact)

# === Portfolio Editor ===
def portfolio_editor():
    st.subheader("4. Portfolio Allocation")
    assets=['Equities','Gold','Oil','Bonds','Cash']
    df_init=pd.DataFrame({'Asset':assets,'Pct':[40,20,20,15,5]})
    df = st.data_editor(df_init,use_container_width=True) if hasattr(st,'data_editor') else st.experimental_data_editor(df_init,use_container_width=True)
    if abs(df['Pct'].sum()-100)>0.1:
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
    regimes, probs = regimes_and_probs()
    specs = define_scenarios()

    # Scenario drivers & correlations
    st.sidebar.header("5. Scenario Drivers & Correlations")
    values, rets, eps_list, pe_list, corr_vals = [], [], [], [], {}
    # === Scenario Drivers & Correlations ===
    st.sidebar.header("5. Scenario Drivers & Correlations")
    # Default scenario inputs per regime
    gdp_defaults = {'Expansion': 3.0, 'Recession': -1.0, 'Stagflation': 1.0, 'Deflation': -0.5}
    rate_defaults = {'Expansion': 0.2, 'Recession': 1.0, 'Stagflation': 0.8, 'Deflation': -0.2}
    share_defaults = {'Expansion': 0.0, 'Recession': 0.02, 'Stagflation': 0.0, 'Deflation': 0.0}
    values, rets, eps_list, pe_list, corr_vals = [], [], [], [], {}
    for reg in regimes:
        with st.sidebar.expander(reg, True):
            # Scenario-specific default inputs
            gdp = st.number_input(f"GDP Growth {reg}%", gdp_defaults.get(reg, 0.0), key=f"gdp{reg}")
            ratec = st.number_input(f"Rate Shock {reg}%", rate_defaults.get(reg, 0.0), key=f"rs{reg}")
            sharec = st.number_input(f"Share Chg {reg}%", share_defaults.get(reg, 0.0), key=f"sc{reg}")
            corr_vals[reg] = st.slider(f"Eq-Gold Corr {reg}", -1.0, 1.0, -0.2, key=f"c{reg}")
        eps_f = eps_proj(eps, gdp, m2, ratec, sharec)(eps, gdp, m2, ratec, sharec)
        pe_f = pe_from_real(rt) * macro_mult(liq, fiscal, geo)
        fair_spx_reg = eps_f * pe_f
        values.append(fair_spx_reg)
        rets.append(fair_spx_reg/spx - 1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)

    weighted_eps = sum(probs[r]/100 * eps_list[i] for i, r in enumerate(regimes))
    weighted_pe  = sum(probs[r]/100 * pe_list[i] for i, r in enumerate(regimes))
    fair_spx = weighted_eps * weighted_pe

    # Display regime valuation table
    dfv = pd.DataFrame({'Regime': regimes, 'Fair SPX': values, 'Return%': rets, 'P%': [probs[r] for r in regimes]})
    st.write(dfv)

    # Valuation & Asset Price Anchors Comparison
    st.subheader("6. Valuation & Asset Price Anchors")
    gold_price = price_gold(rt, st.sidebar.number_input("VIX for Gold", 16), geo_score)
    oil_price = price_oil(
        st.sidebar.number_input("Oil Inv Change (%), 0.0),
        st.sidebar.slider("OPEC Quota", -1.0, 1.0, 0.0)  # Default no shock),
        st.sidebar.number_input("Global PMI", 50.0),
        geo_score
    )
    bond_yield = nelson_siegel(rt)

    anchors = {
        'Metric': ["Current SPX","Weighted EPS","Weighted P/E","Gold","Oil","10Y Yield"],
        'Actual': [spx, eps, spx/eps, a_gold, a_oil, a_10y/100],
        'Model':  [fair_spx, weighted_eps, weighted_pe, gold_price, oil_price, bond_yield]
    }
    df_anchors = pd.DataFrame(anchors)
    st.table(df_anchors.set_index('Metric').style.format({"Actual":"{:.2f}","Model":"{:.2f}"}))

    # Portfolio & Expected Return
    dfp, alloc = portfolio_editor()
    exp_eq = sum(probs[r]/100 * rets[i] for i, r in enumerate(regimes))
    ret_asset = np.array([exp_eq, gold_price/a_gold-1, oil_price/a_oil-1, bond_yield, DEFAULT_CASH_YIELD])
    exp_return = alloc @ ret_asset
    st.subheader("7. Expected Portfolio Return")
    st.metric("Expected Return", f"{exp_return:.2%}")

    # Monte Carlo Simulation & Distribution
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
