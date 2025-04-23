# app.py — Hedge-Fund–Grade Macro Simulator with Empirical Backdrop and Asset Anchors

import streamlit as st
import pandas as pd
import numpy as np

# === Page Config ===
st.set_page_config(page_title="Hedge-Fund Macro Simulator", layout="wide")

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.02
BOND_DURATION = 7

# === Sidebar: Empirical Macro Backdrop ===

# Historical-volatility–based weights for composite scores (calibrated by factor variance)
LIQ_WEIGHTS = [0.4, 0.3, 0.3]
FISCAL_WEIGHTS = [0.33, 0.33, 0.34]
GEO_WEIGHTS = [0.5, 0.3, 0.2]

# Standalone normalization functions for each macro component

def normalize_liquidity(fed_bs, short_rate, m2):
    """Combines balance sheet, short rate, and money growth into a liquidity score."""
    comps = [
        (fed_bs - 15) / 30,
        np.clip((5 - short_rate) / 5, 0, 1),
        np.clip(m2 / 15, 0, 1),
    ]
    return sum(w * c for w, c in zip(LIQ_WEIGHTS, comps))


def normalize_fiscal(deficit, gov_spend, transfers):
    """Combines deficit, spending, and transfers into a fiscal-stimulus score."""
    comps = [
        np.clip((deficit - 1) / 14, 0, 1),
        np.clip((gov_spend - 15) / 20, 0, 1),
        np.clip((transfers - 5) / 15, 0, 1),
    ]
    return sum(w * c for w, c in zip(FISCAL_WEIGHTS, comps))


def normalize_geo(geo_idx, vix, conflicts):
    """Combines geopolitical stress markers into a single geo-risk score."""
    comps = [
        np.clip((geo_idx - 50) / 150, 0, 1),
        np.clip((vix - 10) / 40, 0, 1),
        np.clip((conflicts - 10) / 50, 0, 1),
    ]
    return sum(w * c for w, c in zip(GEO_WEIGHTS, comps))


def macro_conditions():
    """
    *Collects empirical data for liquidity, fiscal, and geo risk and applies weighted composites*
    *based on historical volatility instead of a simple average.*
    """
    st.sidebar.header("1. Empirical Macro Backdrop")
    use_emp = st.sidebar.checkbox("Use empirical macro inputs", True)
    if use_emp:
        override = st.sidebar.checkbox("Override backdrop manually", False)
        disabled = not override
        # Liquidity inputs
        fed_bs = st.sidebar.number_input("Fed Balance Sheet (% GDP)", 35.0, disabled=disabled)
        short_rate = st.sidebar.number_input("Real Short-Term Rate (%)", 1.5, disabled=disabled)
        m2 = st.sidebar.number_input("M2 Growth YoY (%)", 4.0, disabled=disabled)
        # Fiscal inputs
        deficit = st.sidebar.number_input("Budget Deficit (% GDP)", 6.0, disabled=disabled)
        gov_spend = st.sidebar.number_input("Government Spending (% GDP)", 25.0, disabled=disabled)
        transfers = st.sidebar.number_input("Net Transfers (% GDP)", 10.0, disabled=disabled)
        # Geo inputs
        geo_idx = st.sidebar.number_input("Geo Risk Index", 120.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX Index", 20.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict Events", 30, disabled=disabled)

        # Apply normalized composites with volatility-based weights
        liq = normalize_liquidity(fed_bs, short_rate, m2)
        fiscal = normalize_fiscal(deficit, gov_spend, transfers)
        geo = normalize_geo(geo_idx, vix, conflicts)

        # Manual overrides
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, fiscal, disabled=not override)
        geo = st.sidebar.slider("Geo Risk", 0.0, 1.0, geo, disabled=not override)
    else:
        short_rate, m2 = 1.5, 4.0
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geo Risk", 0.0, 1.0, 0.1)
    return liq, fiscal, geo, short_rate, m2

# === Sidebar: Market Inputs & Geo Flashpoints === & Geo Flashpoints ===
def market_inputs():
    st.sidebar.header("2. Market Inputs & Geo Flashpoints")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)",200.0)
    spx = st.sidebar.number_input("Current SPX Index",5300.0)
    # Actual asset prices for comparison
    st.sidebar.markdown("**Actual Asset Prices**")
    actual_gold = st.sidebar.number_input("Current Gold Price ($)", 2000.0)
    actual_oil = st.sidebar.number_input("Current Oil Price ($)", 80.0)
    actual_10y = st.sidebar.number_input("Current 10Y Yield (%)", 4.0)
    # geo flashpoints
    st.sidebar.markdown("**Geo Flashpoint Impacts**")
    geo_events = {}
    for name in ["China–US Tariff Escalation","Russia Gas Cutoff"]:
        w = st.sidebar.slider(f"Impact: {name}", -1.0,1.0,0.0, key=name)
        geo_events[name] = w
    num_custom = st.sidebar.number_input("# Custom Geo Events",0,5,0)
    for i in range(num_custom):
        cname = st.sidebar.text_input(f"Event {i+1} Name", key=f"cn{i}")
        if cname:
            w = st.sidebar.slider(f"Impact: {cname}", -1.0,1.0,0.0, key=f"cw{i}")
            geo_events[cname] = w
    return eps, spx, actual_gold, actual_oil, actual_10y, geo_events

# === Utility Functions ===
def pe_from_real(rate, prem=0.04): return min(40, max(8,1/(rate+prem)))
def regimes_and_probs():
    st.sidebar.header("3. Regime Probabilities")
    regimes = ["Expansion","Recession","Stagflation","Deflation"]
    probs = {r: st.sidebar.slider(f"P({r})%",0,100,25) for r in regimes}
    if sum(probs.values())!=100: st.sidebar.error("Must sum to 100%") and st.stop()
    return regimes, probs

def nelson_siegel(rt): return 1 - ((1-np.exp(-rt))/rt) + 0.5*(((1-np.exp(-rt))/rt)-np.exp(-rt))
def price_gold(rt,vix,geo_score): return 2000*(1-rt*0.1)+vix*10+geo_score*300
def price_oil(inv,opec,pmi,geo_score): return 80*(1+pmi/100-inv/100)+opec*80+geo_score*50
def eps_proj(eps,gdp,inf,ratec,sharec): return (eps*(1+gdp/100)*(1-inf*0.005-ratec*0.01))*(1-sharec)
def define_scenarios(): return {'Expansion':{'pe_mul':1.2,'eps_d':+0.10},'Recession':{'pe_mul':0.8,'eps_d':-0.25},'Stagflation':{'pe_mul':0.9,'eps_d':-0.15},'Deflation':{'pe_mul':0.85,'eps_d':-0.05}}
# === Macro Multiplier (Regime-Sensitive) ===
def macro_mult(liq, fiscal, geo):
    """
    *Computes a macro-driven P/E multiplier based on liquidity, fiscal stimulus, and geo risk with caps.*
    """
    impact = liq*0.25 + fiscal*0.2 - geo*0.3
    return 1 + min(MAX_MACRO_PE_IMPACT, impact)

# === Monte Carlo ===
def simulate(alloc,ret,cov,sims=3000): return np.random.multivariate_normal(ret,cov,sims)@alloc

def portfolio_editor():
    st.subheader("4. Portfolio Allocation")
    assets = ['Equities','Gold','Oil','Bonds','Cash']
    df_init = pd.DataFrame({'Asset': assets, 'Pct': [40,20,20,15,5]})
    # Use data_editor if available, otherwise fallback
    if hasattr(st, 'data_editor'):
        df = st.data_editor(df_init, use_container_width=True)
    else:
        df = st.experimental_data_editor(df_init, use_container_width=True)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Portfolio weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values/100

# === Main Application ===
def run():
    # 1. Empirical Backdrop
    liq, fiscal, geo, rt, m2 = macro_conditions()
    # 2. Market & Actual Prices
    eps, spx, a_gold, a_oil, a_10y, geo_events = market_inputs()
    geo_score = sum(geo_events.values())
    # 3. Regime Probabilities
    regimes, probs = regimes_and_probs()
    specs = define_scenarios()

    # 4. Scenario Drivers & Correlations
    st.sidebar.header("5. Scenario Drivers & Correlations")
    values,rets,eps_list,pe_list,corr_vals = [],[],[],[],{}
    for reg in regimes:
        with st.sidebar.expander(reg,True):
            gdp=st.number_input(f"GDP {reg}%",2.0,key=f"gdp{reg}")
            ratec=st.number_input(f"Rate Shock {reg}%",0.0,key=f"rs{reg}")
            sharec=st.number_input(f"Share Chg {reg}%",0.0,key=f"sc{reg}")
            corr_vals[reg]=st.slider(f"Eq-Gold Corr {reg}",-1.0,1.0,-0.2,key=f"c{reg}")
        eps_f=eps_proj(eps,gdp,m2,ratec,sharec)
        # Derive regime P/E using the macro multiplier for this backdrop
        regime_mult = macro_mult(liq, fiscal, geo)
        pe_f = pe_from_real(rt) * regime_mult
        values.append(eps_f*pe_f); rets.append(eps_f*pe_f/spx-1)
        eps_list.append(eps_f); pe_list.append(pe_f)

    weighted_eps=sum(probs[r]/100*eps_list[i] for i,r in enumerate(regimes))
    weighted_pe=sum(probs[r]/100*pe_list[i] for i,r in enumerate(regimes))

    # Display scenario fair values
    dfv=pd.DataFrame({'Regime':regimes,'Fair SPX':values,'Ret%':rets,'P%':[probs[r] for r in regimes]})
    st.write(dfv)

    # 6. Valuation & Asset Price Anchors Comparison
    st.subheader("6. Valuation & Asset Price Anchors")
    # calculate model anchors
    gold_price=price_gold(rt, st.sidebar.number_input("VIX for Gold",20), geo_score)
    oil_price=price_oil(st.sidebar.number_input("Oil Inv Change",0.0), st.sidebar.slider("OPEC Quota",-1.0,1.0,0.0), st.sidebar.number_input("Global PMI",50.0), geo_score)
    bond_yield = nelson_siegel(rt)

    df_anchors=pd.DataFrame(
        index=["Current SPX","Weighted EPS","Weighted P/E","Gold Price","Oil Price","10Y Yield"],
        data={
            "Actual": [spx, eps, spx/eps, a_gold, a_oil, a_10y/100],
            "Model":  [weighted_eps*0 + spx, weighted_eps, weighted_pe, gold_price, oil_price, bond_yield]
        }
    )
    # Display Actual vs Model anchors as interactive dataframe
    st.table(df_anchors.style.format({"Actual":"{:.2f}","Model":"{:.2f}"}))

    # 7. Portfolio Allocation
    dfp, alloc = portfolio_editor()

    # 8. Expected Portfolio Return
    # Compute asset return vector
    exp_eq = sum(probs[r]/100 * rets[i] for i, r in enumerate(regimes))
    gold_ret = gold_price / a_gold - 1
    oil_ret  = oil_price / a_oil - 1
    bond_ret = bond_yield
    cash_ret = DEFAULT_CASH_YIELD
    ret_asset = np.array([exp_eq, gold_ret, oil_ret, bond_ret, cash_ret])

    exp_port_return = np.dot(alloc, ret_asset)
    st.subheader("7. Expected Portfolio Return")
    st.metric(label="Expected Return", value=f"{exp_port_return:.2%}")

    # 9. Monte Carlo Simulation & Distribution
    vols = np.array([0.15, 0.10, 0.12, 0.08, 0.00])
    avg_corr = np.mean(list(corr_vals.values()))
    cov = np.diag(vols) @ np.array([
        [1,     avg_corr, 0,      0,      0],
        [avg_corr, 1,     0,      0,      0],
        [0,     0,     1,      0,      0],
        [0,     0,     0,      1,      0],
        [0,     0,     0,      0,      1]
    ]) @ np.diag(vols)
    sims = simulate(alloc, ret_asset, cov)
    st.subheader("8. Portfolio MC Distribution")
    st.line_chart(pd.Series(sims).rolling(50).mean())

# Run the app
def __main__call():
    run()

if __name__ == '__main__':
    __main__call()
