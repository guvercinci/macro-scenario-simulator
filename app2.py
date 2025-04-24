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

# === Sidebar Sections ===

def market_prices():
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

# Macro backdrop
LIQ_WEIGHTS=[0.4,0.3,0.3]
FISCAL_WEIGHTS=[0.33,0.33,0.34]
GEO_WEIGHTS=[0.5,0.3,0.2]

def normalize_liquidity(bs,sr,m2):
    comps=[(bs-15)/30, np.clip((5-sr)/5,0,1), np.clip(m2/15,0,1)]
    return sum(w*c for w,c in zip(LIQ_WEIGHTS,comps))
def normalize_fiscal(defi,spend,trans):
    comps=[np.clip((defi-1)/14,0,1), np.clip((spend-15)/20,0,1), np.clip((trans-5)/15,0,1)]
    return sum(w*c for w,c in zip(FISCAL_WEIGHTS,comps))
def normalize_geo(idx,vix,conf):
    comps=[np.clip((idx-50)/150,0,1), np.clip((vix-10)/40,0,1), np.clip((conf-10)/50,0,1)]
    return sum(w*c for w,c in zip(GEO_WEIGHTS,comps))

def macro_backdrop():
    st.sidebar.header("Macro Backdrop")
    use_emp=st.sidebar.checkbox("Use empirical inputs",True)
    if use_emp:
        override=st.sidebar.checkbox("Manual override",False)
        disabled=not override
        fed_bs=st.sidebar.number_input("Fed BS (%GDP)",38.0,disabled=disabled)
        sr=st.sidebar.number_input("Real short rate (%)",2.5,disabled=disabled)
        m2=st.sidebar.number_input("M2 growth YoY (%)",6.0,disabled=disabled)
        defi=st.sidebar.number_input("Budget deficit (%GDP)",5.0,disabled=disabled)
        spend=st.sidebar.number_input("Govt spending (%GDP)",24.0,disabled=disabled)
        trans=st.sidebar.number_input("Transfers (%GDP)",12.0,disabled=disabled)
        idx=st.sidebar.number_input("Geo risk index",100.0,disabled=disabled)
        vix=st.sidebar.number_input("VIX index",16.0,disabled=disabled)
        conf=st.sidebar.number_input("Conflict events (#)",20,disabled=disabled)
        liq=normalize_liquidity(fed_bs,sr,m2)
        fis=normalize_fiscal(defi,spend,trans)
        geo=normalize_geo(idx,vix,conf)
        liq=st.sidebar.slider("Liquidity score",0.0,1.0,liq,disabled=not override)
        fis=st.sidebar.slider("Fiscal score",0.0,1.0,fis,disabled=not override)
        geo=st.sidebar.slider("Geo risk score",0.0,1.0,geo,disabled=not override)
    else:
        sr,m2=2.5,6.0
        liq=st.sidebar.slider("Liquidity score",0.0,1.0,0.5)
        fis=st.sidebar.slider("Fiscal score",0.0,1.0,0.3)
        geo=st.sidebar.slider("Geo risk score",0.0,1.0,0.1)
    return liq,fis,geo,sr,m2

# Regime probabilities
def regime_probs(liq,fis,geo):
    st.sidebar.header("Regime Probabilities")
    scores=[liq*0.6+fis*0.4-geo*0.2,(1-liq)*0.7+geo*0.3,fis*0.2+geo*0.5+liq*0.1,(1-fis)*0.5+(1-geo)*0.5-liq*0.2]
    auto=np.clip(scores,0,None)
    auto=auto/auto.sum()
    regames=['Expansion','Recession','Stagflation','Deflation']
    ov=st.sidebar.checkbox("Override probabilities",False)
    probs={r:st.sidebar.number_input(f"P({r})%",0,100,int(auto[i]*100),disabled=not ov,key=r) for i,r in enumerate(regames)}
    total=sum(probs.values()); st.sidebar.markdown(f"**Total: {total}%**")
    if ov and total!=100: st.sidebar.error("Sum must be 100%.") and st.stop()
    if not ov:
        dif=100-sum(int(auto[i]*100) for i in range(4)); probs[regames[np.argmax(auto)]]+=dif
    return regames,probs

# Portfolio allocation
def portfolio_allocation():
    st.subheader("Portfolio Allocation")
    beta=st.sidebar.slider("Equity Beta",0.0,2.0,1.0,step=0.1)
    df=pd.DataFrame({'Asset':['Equities','Gold','Oil','Bonds','Cash'],'Pct':[40,20,20,15,5]})
    df=st.data_editor(df,use_container_width=True) if hasattr(st,'data_editor') else st.experimental_data_editor(df,use_container_width=True)
    if abs(df['Pct'].sum()-100)>0.1: st.error("Weights must sum to 100%.") and st.stop()
    return df,df['Pct']/100,beta

# Scenario drivers

def scenario_drivers(eps,spx,sr,m2,liq,fis,geo,regimes,probs):
    st.sidebar.header("Scenario Drivers & Correlations")
    gdp_def,rate_def,share_def={'Expansion':3,'Recession':-1,'Stagflation':1,'Deflation':-0.5},{'Expansion':0.2,'Recession':1,'Stagflation':0.8,'Deflation':-0.2},{'Expansion':0,'Recession':0.02,'Stagflation':0,'Deflation':0}
    vals,rets,eps_list,pe_list,corrs=[],[],[],[],{}
    for r in regimes:
        with st.sidebar.expander(r,True):
            g=st.number_input(f"GDP {r}%",gdp_def[r],key=f"gdp{r}")
            rc=st.number_input(f"Rate shock {r}%",rate_def[r],key=f"rc{r}")
            sc=st.number_input(f"Share change {r}%",share_def[r],key=f"sc{r}")
            corrs[r]=st.sidebar.slider(f"Eq-Gold corr {r}",-1,1,-0.2,key=f"c{r}")
        eps_f=eps*(1+g/100)*(1-0.005*m2-0.01*rc)*(1-sc)
        pe=min(40,max(8,1/((sr/100)+0.04)))*(1+min(MAX_MACRO_PE_IMPACT,liq*0.25+fis*0.2-geo*0.3))
        fair=eps_f*pe
        vals.append(fair); rets.append(fair/spx-1)
        eps_list.append(eps_f); pe_list.append(pe)
    return vals,rets,eps_list,pe_list,corrs

# Anchors inputs

def anchor_inputs(): return (st.sidebar.number_input("VIX for Gold",16.0),st.sidebar.number_input("Oil inv change (%)",0.0),st.sidebar.slider("OPEC quota",-1,1,0.0),st.sidebar.number_input("Global PMI",50.0))

# Helpers

def nelson_siegel(x): return 1-((1-np.exp(-x))/x)+0.5*(((1-np.exp(-x))/x)-np.exp(-x))

# === Main ===
def run():
    eps,spx,a_gold,a_oil,a_10y,geo=market_prices()
    liq,fis,geo_r,sr,m2=macro_backdrop()
    regimes,probs=regime_probs(liq,fis,geo_r)
    df,alloc,beta=portfolio_allocation()
    vals,rets,epsl,pel,corrs=scenario_drivers(eps,spx,sr,m2,liq,fis,geo_r,regimes,probs)

    # Fair-value table
    dfv=pd.DataFrame({'Regime':regimes,'Fair SPX':[f"${v:,.0f}" for v in vals],'Return%':[f"{r:.1%}" for r in rets],'P%':[f"{probs[r]:.1f}%" for r in regimes]})
    st.subheader("Regime Fair-Value Table")
    st.table(dfv.set_index('Regime'))

    # Anchors
    vix,inv,opec,pmi=anchor_inputs()
    avg_corr=sum(probs[r]/100*corrs[r] for r in regimes)
    gold_price=2000*(1-sr*0.1)+vix*10+sum(geo.values())*300-avg_corr*200
    oil_price=80*(1+pmi/100-inv/100)+opec*80+sum(geo.values())*50
    bond_y=nelson_siegel(sr)
    metrics=["SPX","EPS","P/E","Gold","Oil","10Y"]
    actual=[f"${spx:,.0f}",f"${eps:,.0f}",f"{(spx/eps):.1f}",f"${a_gold:,.0f}",f"${a_oil:,.0f}",f"{a_10y:.1%}"]
    model=[f"${vals[0]:,.0f}",f"${epsl[0]:,.0f}",f"{pel[0]:.1f}",f"${gold_price:,.0f}",f"${oil_price:,.0f}",f"{bond_y:.1%}"]
    anc=pd.DataFrame({'Actual':actual,'Model':model},index=metrics)
    st.subheader("Valuation & Asset Price Anchors")
    st.table(anc)

    # Expected return
    exp_eq=sum(probs[r]/100*rets[i] for i,r in enumerate(regimes))*beta
    ret_asset=np.array([exp_eq,(gold_price/a_gold-1),(oil_price/a_oil-1),bond_y,DEFAULT_CASH_YIELD])
    exp_ret=(alloc*ret_asset).sum()
    st.subheader("Expected Portfolio Return")
    st.metric("Return",f"{exp_ret:.2%}")

    # MC
    vols=np.array([0.15,0.1,0.12,0.08,0.0])
    cov=np.diag(vols)@np.array([[1,sum(corrs.values())/4,0,0,0],[sum(corrs.values())/4,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])@np.diag(vols)
    sims=np.random.multivariate_normal(ret_asset,cov,3000)@(alloc)
    st.subheader("Portfolio MC Distribution")
    st.line_chart(pd.Series(sims).rolling(50).mean())

if __name__=='__main__':
    run()
