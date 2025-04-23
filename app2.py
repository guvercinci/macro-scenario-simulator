# app.py — Enhanced & Empirical Streamlit Macro Portfolio Simulator
# Combines data-driven empirical inputs with advanced macro modeling with robust editor fallback

import streamlit as st
import pandas as pd
import numpy as np

# === Page Configuration ===
st.set_page_config(
    page_title="Macro Portfolio Simulator",
    layout="wide",
)

# === Constants & Defaults ===
MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.04
BOND_DURATION = 7

# === Sidebar: Valuation Inputs ===
def valuation_inputs():
    st.sidebar.header("1. Market Inputs")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", value=200.0)
    spx = st.sidebar.number_input("Current SPX Index", value=5300.0)
    return eps, spx

# === Sidebar: Macro Conditions (Empirical + Override) ===
def macro_conditions():
    st.sidebar.header("2. Macro Backdrop")
    use_empirical = st.sidebar.checkbox("Use empirical macro inputs", value=True)
    override = st.sidebar.checkbox("Override derived backdrop manually", value=False)
    disabled = not override

    # set defaults
    short_term_rate = 1.5
    m2_growth = 4.0

    if use_empirical:
        # Liquidity
        st.sidebar.markdown("#### Empirical Inputs: Liquidity")
        fed_bs = st.sidebar.number_input("Fed Balance Sheet (% GDP)", value=35.0, disabled=disabled)
        short_term_rate = st.sidebar.number_input("Real Short-Term Rate (%)", value=1.5, disabled=disabled)
        m2_growth = st.sidebar.number_input("M2 Growth YoY (%)", value=4.0, disabled=disabled)

        # Fiscal
        st.sidebar.markdown("#### Empirical Inputs: Fiscal")
        deficit = st.sidebar.number_input("Budget Deficit (% GDP)", value=6.0, disabled=disabled)
        gov_spend = st.sidebar.number_input("Government Spending (% GDP)", value=25.0, disabled=disabled)
        transfers = st.sidebar.number_input("Net Transfers (% GDP)", value=10.0, disabled=disabled)

        # Geopolitical
        st.sidebar.markdown("#### Empirical Inputs: Geopolitical Risk")
        geo_index = st.sidebar.number_input("Geo Risk Index", value=120.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX Index", value=20.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict Events", value=30, disabled=disabled)

        # normalize
        liq = np.mean([
            (fed_bs - 15) / 30,
            np.clip((5 - short_term_rate) / 5, 0, 1),
            np.clip(m2_growth / 15, 0, 1)
        ])
        fiscal = np.mean([np.clip((deficit - 1) / 14,0,1), np.clip((gov_spend - 15)/20,0,1), np.clip((transfers-5)/15,0,1)])
        geo = np.mean([np.clip((geo_index-50)/150,0,1), np.clip((vix-10)/40,0,1), np.clip((conflicts-10)/50,0,1)])

        # override sliders
        st.sidebar.markdown("#### Manual Overrides")
        liq = st.sidebar.slider("Liquidity", 0.0,1.0,liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0,1.0,fiscal, disabled=not override)
        geo = st.sidebar.slider("Geopolitical Risk", 0.0,1.0,geo, disabled=not override)
    else:
        liq = st.sidebar.slider("Liquidity",0.0,1.0,0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus",0.0,1.0,0.3)
        geo = st.sidebar.slider("Geopolitical Risk",0.0,1.0,0.1)

    return liq, fiscal, geo, short_term_rate, m2_growth

# === PE from Real Rates ===
def pe_from_real(real_rate, eq_premium=0.04):
    req = real_rate + eq_premium
    return min(40, max(8, 1/req))

# === Default Scenarios ===
def default_scenarios(real_rate):
    return {
        "Boom": {"pe": pe_from_real(1.0), "eps_ch": +0.15},
        "Recession": {"pe": pe_from_real(2.5), "eps_ch": -0.20},
        "Stagflation": {"pe": pe_from_real(2.8), "eps_ch": -0.10},
        "Deflation": {"pe": pe_from_real(1.8), "eps_ch": -0.05},
    }

# === Macro Multiplier ===
def macro_mult(liq, fiscal, geo):
    impact = liq*0.25 + fiscal*0.2 - geo*0.3
    return 1 + min(MAX_MACRO_PE_IMPACT, impact)

# === Asset Price Anchors ===
def macro_targets(liq, fiscal, geo, short_term_rate, m2_growth):
    gold = 2000*(1 + max(0,(3-short_term_rate))*0.25 + max(0,(m2_growth-5))*0.1 + geo*0.4)
    crude = 80*(1 + fiscal*0.2 + geo*0.3 + liq*0.1)
    bond = 4*(1 - liq*0.05 - geo*0.05 + fiscal*0.05)
    return {"Gold":gold, "Crude":crude, "10Y":bond}

# === Scenario Probabilities ===
def scenario_probs():
    st.sidebar.header("3. Scenario Probabilities")
    probs,total={},0
    for stmt in ["Boom","Recession","Stagflation","Deflation"]:
        p=st.sidebar.slider(f"{stmt} (%)",0,100,25)
        probs[stmt]=p; total+=p
    st.sidebar.markdown(f"**Total: {total}%**")
    if total!=100: st.sidebar.error("Sum must 100%") and st.stop()
    return probs

# === Portfolio Editor w/ Fallback ===
def portfolio_editor():
    st.subheader("4. Portfolio Allocation")
    size=st.number_input("Portfolio Size ($)",100000,step=10000)
    beta=st.sidebar.slider("Equity Beta",0.5,2.0,1.0,step=0.1)
    preset=st.selectbox("Preset",["Balanced","Aggressive","Defensive","Custom"])
    presets={"Balanced":[60,30,5,3,2,0],"Aggressive":[80,10,2,5,2,1],"Defensive":[30,30,10,10,10,10],"Custom":[50,30,10,5,3,2]}
    assets=["Equities","Fixed Income","Cash","Commodities","Gold","Hedging"]
    df_init=pd.DataFrame({"Asset":assets,"Pct":presets[preset]})
    # fallback for editor
    if hasattr(st, 'data_editor'):
        df=st.data_editor(df_init,use_container_width=True)
    else:
        df=st.experimental_data_editor(df_init,use_container_width=True)
    if abs(df['Pct'].sum()-100)>0.1: st.error("Allocation must sum to 100%") and st.stop()
    df['Alloc']=df['Pct']/100*size
    df['Beta']=df['Asset'].apply(lambda x:beta if x=="Equities" else 1)
    return df

# === Fair Value Calculation ===
def calc_fair(eps,scen,probs,liq,fiscal,geo):
    w_eps=sum(probs[s]/100*eps*(1+scen[s]['eps_ch']) for s in scen)
    w_pe =sum(probs[s]/100*scen[s]['pe'] for s in scen)
    m   =macro_mult(liq,fiscal,geo)
    return w_eps,w_pe,m,w_eps*w_pe*m

# === Monte Carlo Simulation ===
def simulate(mu,cov,n=2000): return np.random.multivariate_normal(mu,cov,n)

# === App Entry Point ===
def run():
    # gather inputs
    eps,spx=valuation_inputs()
    liq,fiscal,geo,short_term_rate,m2=macro_conditions()
    probs=scenario_probs()
    scen=default_scenarios(short_term_rate)

    # calc fair
    w_eps,w_pe,mult,fair_spx=calc_fair(eps,scen,probs,liq,fiscal,geo)
    targets=macro_targets(liq,fiscal,geo,short_term_rate,m2)

    # display
    st.title("Macro Portfolio Simulator")
    st.markdown(f"**Fair SPX:** {fair_spx:,.0f}")
    st.markdown(f"W. EPS: {w_eps:.1f}, W. PE: {w_pe:.1f}, Macro Mult: {mult:.2f}")

    # portfolio
    df=portfolio_editor()

    # simulate returns
    mu=np.array([(targets['Gold']/2000-1),(targets['Crude']/80-1),(targets['10Y']/4-1)])
    sigma=np.diag([0.10,0.15,0.05]); cov=sigma@np.eye(3)@sigma
    eq,gold,bond=simulate(mu,cov).T; cash=DEFAULT_CASH_YIELD

    # portfolio returns
    weights=df.set_index('Asset')['Pct']/100
    port=weights['Equities']*eq+weights['Gold']*gold+weights['Fixed Income']*bond+weights['Cash']*cash

    # show distribution
    st.subheader("MC Return Distribution")
    hist=np.histogram(port,bins=50)[0]
    st.bar_chart(hist)

if __name__=="__main__": run()
