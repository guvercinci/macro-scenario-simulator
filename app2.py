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

# === Step 1: Market Prices & Flashpoints ===
def step1_market():
    st.sidebar.header("Market Prices & Flashpoints")
    eps = st.sidebar.number_input("SPX trailing EPS", 220.0, min_value=0.0)
    spx = st.sidebar.number_input("SPX index level", 4500.0, min_value=0.0)
    st.sidebar.markdown("**Current Asset Prices**")
    a_gold = st.sidebar.number_input("Gold price ($)", 1900.0, min_value=0.0)
    a_oil = st.sidebar.number_input("Oil price ($)", 75.0, min_value=0.0)
    a_10y = st.sidebar.number_input("10Y yield (%)", 3.5, min_value=0.0)
    st.sidebar.markdown("**Geo Flashpoints**")
    geo_events = {name: st.sidebar.slider(f"Impact: {name}", -1.0, 1.0, 0.0) 
                  for name in ["Tariff shock", "Gas cutoff"]}
    return eps, spx, a_gold, a_oil, a_10y, geo_events

# === Step 2: Macro Backdrop ===
LIQ_WEIGHTS = [0.4, 0.3, 0.3]
FISCAL_WEIGHTS = [0.33, 0.33, 0.34]
GEO_WEIGHTS = [0.5, 0.3, 0.2]

def normalize_liquidity(bs, sr, m2):
    comps = [(bs - 15)/30, np.clip((5-sr)/5,0,1), np.clip(m2/15,0,1)]
    return sum(w*c for w,c in zip(LIQ_WEIGHTS,comps))

def normalize_fiscal(defi, spend, transf):
    comps = [np.clip((defi-1)/14,0,1), np.clip((spend-15)/20,0,1), np.clip((transf-5)/15,0,1)]
    return sum(w*c for w,c in zip(FISCAL_WEIGHTS,comps))

def normalize_geo(idx, vix, conf):
    comps = [np.clip((idx-50)/150,0,1), np.clip((vix-10)/40,0,1), np.clip((conf-10)/50,0,1)]
    return sum(w*c for w,c in zip(GEO_WEIGHTS,comps))

def step2_backdrop():
    st.sidebar.header("Macro Backdrop")
    use_emp = st.sidebar.checkbox("Use empirical inputs", True)
    if use_emp:
        override = st.sidebar.checkbox("Manual override", False)
        disabled = not override
        fed_bs = st.sidebar.number_input("Fed BS (% GDP)", 38.0, disabled=disabled)
        short_rate = st.sidebar.number_input("Real short rate (%)", 2.5, disabled=disabled)
        m2 = st.sidebar.number_input("M2 growth YoY (%)", 6.0, disabled=disabled)
        deficit = st.sidebar.number_input("Budget deficit (% GDP)", 5.0, disabled=disabled)
        gov_spend = st.sidebar.number_input("Govt spending (% GDP)", 24.0, disabled=disabled)
        transfers = st.sidebar.number_input("Transfers (% GDP)", 12.0, disabled=disabled)
        geo_idx = st.sidebar.number_input("Geo risk index", 100.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX index", 16.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict events (#)", 20, disabled=disabled)
        liq = normalize_liquidity(fed_bs, short_rate, m2)
        fiscal = normalize_fiscal(deficit, gov_spend, transfers)
        geo = normalize_geo(geo_idx, vix, conflicts)
        liq = st.sidebar.slider("Liquidity score",0.0,1.0,liq,disabled=not override)
        fiscal = st.sidebar.slider("Fiscal score",0.0,1.0,fiscal,disabled=not override)
        geo = st.sidebar.slider("Geo risk score",0.0,1.0,geo,disabled=not override)
    else:
        short_rate, m2 = 2.5,6.0
        liq = st.sidebar.slider("Liquidity score",0.0,1.0,0.5)
        fiscal = st.sidebar.slider("Fiscal score",0.0,1.0,0.3)
        geo = st.sidebar.slider("Geo risk score",0.0,1.0,0.1)
    return liq, fiscal, geo, short_rate, m2

# === Step 3: Regime Probabilities ===
def step3_regimes(liq,fiscal,geo):
    st.sidebar.header("Regime Probabilities")
    exp_s = liq*0.6 + fiscal*0.4 - geo*0.2
    rec_s = (1-liq)*0.7 + geo*0.3
    stag_s = fiscal*0.2 + geo*0.5 + liq*0.1
    def_s = (1-fiscal)*0.5 + (1-geo)*0.5 - liq*0.2
    scores = np.array([exp_s,rec_s,stag_s,def_s])
    probs_auto = (scores.clip(min=0))/scores.clip(min=0).sum()
    regimes = ['Expansion','Recession','Stagflation','Deflation']
    override = st.sidebar.checkbox("Override probabilities",False)
    probs = {}
    for i,r in enumerate(regimes):
        default = int(probs_auto[i]*100)
        probs[r] = st.sidebar.number_input(f"P({r})%",0,100,default,disabled=not override,key=r)
    total = sum(probs.values())
    st.sidebar.markdown(f"**Total: {total}%**")
    if override and total!=100:
        st.sidebar.error("Must sum to 100% when overriding.")
        st.stop()
    if not override:
        diff = 100 - sum(int(probs_auto[i]*100) for i in range(4))
        probs[regimes[np.argmax(probs_auto)]] += diff
    return regimes,probs

# === Step 4: Portfolio Allocation ===
def portfolio_editor():
    st.subheader("Portfolio Allocation")
    equity_beta = st.sidebar.slider("Equity Beta",0.0,2.0,1.0,step=0.1)
    df = pd.DataFrame({'Asset':['Equities','Gold','Oil','Bonds','Cash'],'Pct':[40,20,20,15,5]})
    df = st.data_editor(df,use_container_width=True) if hasattr(st,'data_editor') else st.experimental_data_editor(df,use_container_width=True)
    if abs(df['Pct'].sum()-100)>0.1:
        st.error("Weights must sum to 100%.")
        st.stop()
    return df,df['Pct']/100,equity_beta

# === Step 5: Scenario Drivers & Correlations ===
def step5_drivers(eps,spx,rt,m2,liq,fiscal,geo,regimes,probs):
    st.sidebar.header("Scenario Drivers & Correlations")
    gdp_def={'Expansion':3.0,'Recession':-1.0,'Stagflation':1.0,'Deflation':-0.5}
    rate_def={'Expansion':0.2,'Recession':1.0,'Stagflation':0.8,'Deflation':-0.2}
    share_def={'Expansion':0.0,'Recession':0.02,'Stagflation':0.0,'Deflation':0.0}
    values,rets,eps_list,pe_list,corr_vals=[],[],[],[],{}
    for r in regimes:
        with st.sidebar.expander(r,True):
            gdp=st.number_input(f"GDP {r}%",gdp_def[r],key=f"gdp{r}")
            ratec=st.number_input(f"Rate shock {r}%",rate_def[r],key=f"rs{r}")
            sharec=st.number_input(f"Share change {r}%",share_def[r],key=f"sc{r}")
            corr_vals[r]=st.sidebar.slider(f"Eq-Gold corr {r}",-1.0,1.0,-0.2,key=f"corr{r}")
        eps_f=eps*(1+gdp/100)*(1-0.005*m2-0.01*ratec)*(1-sharec)
        pe_f=min(40,max(8,1/((rt/100)+0.04)))*(1+min(MAX_MACRO_PE_IMPACT,liq*0.25+fiscal*0.2-geo*0.3))
        fair=eps_f*pe_f
        values.append(fair)
        rets.append(fair/spx-1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)
    return values,rets,eps_list,pe_list,corr_vals

# === Step 6: Valuation & Asset Price Anchors ===
def step6_anchors_inputs():
    st.sidebar.header("Anchor Drivers & Assumptions")
    return (st.sidebar.number_input("VIX for Gold",16.0),
            st.sidebar.number_input("Oil inv change (%)",0.0),
            st.sidebar.slider("OPEC quota",-1.0,1.0,0.0),
            st.sidebar.number_input("Global PMI",50.0))

# === Helpers ===
def nelson_siegel(rt): return 1-((1-np.exp(-rt))/rt)+0.5*(((1-np.exp(-rt))/rt)-np.exp(-rt))

def run():
    eps,spx,a_gold,a_oil,a_10y,geo=step1_market()
    liq,fiscal,geo_r,rt,m2=step2_backdrop()
    regimes,probs=step3_regimes(liq,fiscal,geo_r)
    df,alloc,beta=portfolio_editor()
    vals,rets,epsl,pel,corrs=step5_drivers(eps,spx,rt,m2,liq,fiscal,geo_r,regimes,probs)
    # regime table
    dfv=pd.DataFrame({'Regime':regimes,'Fair SPX':vals,'Return%':[f"{x:.1%}" for x in rets],'P%':[f"{probs[r]:.1f}%" for r in regimes]})
    st.write(dfv)
    # anchors
    vix,inv,opec,pmi=step6_anchors_inputs()
    avg_corr=sum(probs[r]/100*corrs[r] for r in regimes)
    gold_price=2000*(1-rt*0.1)+vix*10+sum(geo.values())*300-avg_corr*200
    oil_price=80*(1+pmi/100-inv/100)+opec*80+sum(geo.values())*50
    bond_y=nelson_siegel(rt)
    anc=pd.DataFrame(index=["SPX","EPS","P/E","Gold","Oil","10Y"],data={"Actual":[f"${spx:,.0f}",f"${eps:,.0f}",f"{(spx/eps):.1f}",f"${a_gold:,.0f}",f"${a_oil:,.0f}",f"{a_10y:.1%}"],"Model":[f"${int(epsl[i]*pel[i]):,0f}" if i<1 else None for i in range(6)]})
    st.subheader("Valuation Anchors")
    st.table(anc)
if __name__=='__main__': run()
