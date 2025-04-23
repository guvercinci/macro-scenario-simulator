# app.py — Hedge-Fund–Grade Macro Simulator with Empirical Backdrop
# Final corrected version

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
def macro_conditions():
    """
    *Collects and normalizes empirical data on liquidity, fiscal policy, and geopolitical risk,*
    *then allows manual override.*
    """
    st.sidebar.header("1. Empirical Macro Backdrop")
    use_emp = st.sidebar.checkbox("Use empirical macro inputs", True)
    if use_emp:
        override = st.sidebar.checkbox("Override backdrop manually", False)
        disabled = not override
        # Liquidity
        fed_bs = st.sidebar.number_input("Fed Balance Sheet (% GDP)", 35.0, disabled=disabled)
        short_rate = st.sidebar.number_input("Real Short-Term Rate (%)", 1.5, disabled=disabled)
        m2 = st.sidebar.number_input("M2 Growth YoY (%)", 4.0, disabled=disabled)
        # Fiscal
        deficit = st.sidebar.number_input("Budget Deficit (% GDP)", 6.0, disabled=disabled)
        gov_spend = st.sidebar.number_input("Government Spending (% GDP)", 25.0, disabled=disabled)
        transfers = st.sidebar.number_input("Net Transfers (% GDP)", 10.0, disabled=disabled)
        # Geo Risk
        geo_idx = st.sidebar.number_input("Geo Risk Index", 120.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX Index", 20.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict Events", 30, disabled=disabled)
        # normalize
        liq = np.mean([(fed_bs-15)/30,
                       np.clip((5-short_rate)/5,0,1),
                       np.clip(m2/15,0,1)])
        fiscal = np.mean([np.clip((deficit-1)/14,0,1),
                           np.clip((gov_spend-15)/20,0,1),
                           np.clip((transfers-5)/15,0,1)])
        geo = np.mean([np.clip((geo_idx-50)/150,0,1),
                        np.clip((vix-10)/40,0,1),
                        np.clip((conflicts-10)/50,0,1)])
        # overrides
        liq = st.sidebar.slider("Liquidity",0.0,1.0,liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal Stimulus",0.0,1.0,fiscal, disabled=not override)
        geo = st.sidebar.slider("Geo Risk",0.0,1.0,geo, disabled=not override)
    else:
        short_rate, m2 = 1.5, 4.0
        liq = st.sidebar.slider("Liquidity",0.0,1.0,0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus",0.0,1.0,0.3)
        geo = st.sidebar.slider("Geo Risk",0.0,1.0,0.1)
    return liq, fiscal, geo, short_rate, m2

# === Sidebar: Market Inputs & Flashpoints ===
def market_inputs():
    """
    *Gets SPX EPS/index and allows custom geo-flashpoint events with impact weights.*
    """
    st.sidebar.header("2. Market Inputs & Geo Flashpoints")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)",200.0)
    spx = st.sidebar.number_input("Current SPX Index",5300.0)
    # geo events
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
    return eps, spx, geo_events

# === P/E from Real Rates ===
def pe_from_real(rate, prem=0.04):
    req = rate + prem
    return min(40, max(8,1/req))

# === Scenario Regimes & Probs ===
def regimes_and_probs():
    st.sidebar.header("3. Regime Probabilities")
    regimes = ["Expansion","Recession","Stagflation","Deflation"]
    probs = {r: st.sidebar.slider(f"P({r})%",0,100,25) for r in regimes}
    if sum(probs.values())!=100:
        st.sidebar.error("Sum must 100%")
        st.stop()
    return regimes, probs

# === Yield Curve (Nelson-Siegel) ===
def nelson_siegel(rt):
    lvl=1; slope=-((1-np.exp(-rt))/rt); curv=0.5*(((1-np.exp(-rt))/rt)-np.exp(-rt))
    return lvl+slope+curv

# === Gold & Oil Pricing ===
def price_gold(rt, vix, geo_score):
    return 2000*(1 - rt*0.1) + vix*10 + geo_score*300

def price_oil(inv, opec, pmi, geo_score):
    return 80*(1 + pmi/100 - inv/100) + opec*80 + geo_score*50

# === EPS Projection ===
def eps_proj(eps, gdp, inflation, rate_chg, share_chg):
    rev=eps*(1+gdp/100)
    marg=rev*(1-inflation*0.005-rate_chg*0.01)
    return marg*(1-share_chg)

# === Scenario Specs ===
def define_scenarios():
    return {
        'Expansion':{'pe_mul':1.2,'eps_d':+0.10},
        'Recession':{'pe_mul':0.8,'eps_d':-0.25},
        'Stagflation':{'pe_mul':0.9,'eps_d':-0.15},
        'Deflation':{'pe_mul':0.85,'eps_d':-0.05}
    }

# === Portfolio Editor ===
def portfolio_editor():
    st.subheader("4. Portfolio Allocation")
    assets=['Equities','Gold','Oil','Bonds','Cash']
    df=pd.DataFrame({'Asset':assets,'Pct':[40,20,20,15,5]})
    df=st.data_editor(df,use_container_width=True)
    if abs(df['Pct'].sum()-100)>0.1:
        st.error("Sum !=100%")
        st.stop()
    return df, df['Pct'].values/100

# === Simulation ===
def simulate(alloc, ret, cov, sims=3000):
    draws=np.random.multivariate_normal(ret,cov,sims)
    return (draws*alloc).sum(axis=1)

# === Main ===
def run():
    liq,fiscal,geo_norm,rt,m2=macro_conditions()
    eps,spx,geo_events=market_inputs()
    geo_score=sum(geo_events.values())
    regimes,probs=regimes_and_probs()
    specs=define_scenarios()

    # scenario inputs grouped
    st.sidebar.header("5. Scenario Drivers & Correlations")
    values,rets=[],[]; corr_vals={}
    for reg in regimes:
        with st.sidebar.expander(reg,expanded=True):
            gdp=st.number_input(f"GDP {reg}%",2.0,key=f"gdp{reg}")
            rate_chg=st.number_input(f"Rate Shock {reg}%",0.0,key=f"rs{reg}")
            share_chg=st.number_input(f"Share Chg {reg}%",0.0,key=f"sc{reg}")
            corr_vals[reg]=st.slider(f"Eq-Gold Corr {reg}",-1.0,1.0,-0.2,key=f"c{reg}")
        eps_f=eps_proj(eps,gdp,m2,rate_chg,share_chg)
        pe=pe_from_real(rt)*specs[reg]['pe_mul']
        fair=eps_f*pe
        values.append(fair); rets.append(fair/spx-1)
    dfv=pd.DataFrame({'Regime':regimes,'Fair':values,'Ret':rets,'P': [probs[r] for r in regimes]})
    st.write(dfv)

    # Assets anchors
    vix=st.sidebar.number_input("VIX",20)
    inv=st.sidebar.number_input("Oil Inv%",0.0)
    opec=st.sidebar.slider("OPEC Quota",-1.0,1.0,0.0)
    gold=price_gold(rt,vix,geo_score)
    oil=price_oil(inv,opec,st.sidebar.number_input("Global PMI",50.0),geo_score)
    bond=nelson_siegel(rt)*0.01

    dfp,alloc=portfolio_editor()
    exp_eq=sum(probs[r]/100*rets[i] for i,r in enumerate(regimes))
    ret_asset=np.array([exp_eq, gold/2000-1, oil/80-1, bond, DEFAULT_CASH_YIELD])
    vols=np.array([0.15,0.10,0.12,0.08,0.00])
    avg_corr=np.mean(list(corr_vals.values()))
    cov=np.diag(vols)@np.array([[1,avg_corr,0,0,0],[avg_corr,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])@np.diag(vols)
    sims=simulate(alloc,ret_asset,cov)
    st.subheader("Portfolio MC Distribution")
    st.markdown("_Shows the simulated distribution of portfolio returns over many Monte Carlo runs, highlighting expected performance and tail risk._")
    st.line_chart(pd.Series(sims).rolling(50).mean())

if __name__=='__main__': run()
