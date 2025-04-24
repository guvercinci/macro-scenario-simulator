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
    st.sidebar.markdown("*Enter asset prices and geo-event impacts.*")
    eps = st.sidebar.number_input("SPX trailing EPS", value=220.0, min_value=0.0)
    spx = st.sidebar.number_input("SPX index level", value=4500.0, min_value=0.0)
    st.sidebar.markdown("**Current Asset Prices**")
    a_gold = st.sidebar.number_input("Gold price ($)", value=1900.0, min_value=0.0)
    a_oil = st.sidebar.number_input("Oil price ($)", value=75.0, min_value=0.0)
    a_10y = st.sidebar.number_input("10Y yield (%)", value=3.5, min_value=0.0)
    st.sidebar.markdown("**Geo Flashpoints**")
    geo_events = {name: st.sidebar.slider(f"Impact: {name}", -1.0, 1.0, 0.0)
                  for name in ["Tariff shock", "Gas cutoff"]}
    return eps, spx, a_gold, a_oil, a_10y, geo_events

# === Step 2: Macro Backdrop ===
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

# === Step 3: Regime Probabilities ===
def step3_regimes(liq, fiscal, geo):
    st.sidebar.header("Regime Probabilities")
    st.sidebar.markdown("*Auto-computed from backdrop, optional override.*")
    exp_p = max(liq * 0.6 + fiscal * 0.4 - geo * 0.2, 0)
    rec_p = max((1 - liq) * 0.7 + geo * 0.3, 0)
    stag_p = max(fiscal * 0.2 + geo * 0.5 + liq * 0.1, 0)
    defl_p = max((1 - fiscal) * 0.5 + (1 - geo) * 0.5 - liq * 0.2, 0)
    total = exp_p + rec_p + stag_p + defl_p
    auto = {'Expansion': exp_p/total, 'Recession': rec_p/total,
            'Stagflation': stag_p/total, 'Deflation': defl_p/total}
    override = st.sidebar.checkbox("Override probabilities", False)
    probs = {}
    for r, pct in auto.items():
        default = int(pct * 100)
        probs[r] = st.sidebar.number_input(f"P({r})%", min_value=0, max_value=100,
                                            value=default, disabled=not override, key=f"prob_{r}")
    if not override:
        vals = [int(auto[r]*100) for r in auto]
        diff = 100 - sum(vals)
        idx = vals.index(max(vals))
        vals[idx] += diff
        probs = dict(zip(auto.keys(), vals))
    return list(auto.keys()), probs

# === Step 4: Portfolio Allocation ===
def portfolio_editor():
    st.subheader("Portfolio Allocation")
    col1, col2 = st.columns([3,1])
    with col1:
        df_init = pd.DataFrame({'Asset':['Equities','Gold','Oil','Bonds','Cash'], 'Pct':[40,20,20,15,5]})
        df = st.data_editor(df_init, use_container_width=True) if hasattr(st,'data_editor') else st.experimental_data_editor(df_init, use_container_width=True)
    with col2:
        equity_beta = st.number_input("Equity Beta", value=1.0, min_value=0.0, step=0.1)
    if abs(df['Pct'].sum()-100)>0.1:
        st.error("Weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values/100, equity_beta

# === Step 5: Scenario Drivers & Valuation ===
def step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs):
    st.sidebar.header("Scenario Drivers & Correlations")
    gdp_def = {'Expansion':3.0,'Recession':-1.0,'Stagflation':1.0,'Deflation':-0.5}
    rate_def = {'Expansion':0.2,'Recession':1.0,'Stagflation':0.8,'Deflation':-0.2}
    share_def= {'Expansion':0.0,'Recession':0.02,'Stagflation':0.0,'Deflation':0.0}
    values, rets, eps_list, pe_list, corrs = [],[],[],[],{}
    for r in regimes:
        with st.sidebar.expander(r, True):
            g = st.number_input(f"GDP {r}%", gdp_def[r], key=f"gdp_{r}")
            rc=st.number_input(f"Rate shock {r}%", rate_def[r], key=f"rs_{r}")
            sc=st.number_input(f"Share change {r}%", share_def[r], key=f"sc_{r}")
            corrs[r]=st.slider(f"Eq-Gold corr {r}",-1.0,1.0,-0.2,key=f"corr_{r}")
        eps_proj = eps*(1+g/100)*(1 - m2*0.005 - rc*0.01) - rc*0.1
        eps_floor = eps*0.125
        eps_f = max(eps_proj, eps_floor)*(1-sc)
        pe_base = 1/((rt/100)+0.04)
        pe_f = min(40, max(8, pe_base)) * (1 + min(MAX_MACRO_PE_IMPACT, liq*0.25+fiscal*0.2-geo*0.3))
        fair_v = eps_f * pe_f
        values.append(fair_v)
        rets.append(fair_v/spx - 1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)
    return values, rets, eps_list, pe_list, corrs

# === Step 6: Anchor Drivers & Assumptions ===
def step6_anchors_inputs():
    st.sidebar.header("Anchor Drivers & Assumptions")
    st.sidebar.markdown("*Inputs for valuation & asset price anchors.*")
    vix = st.sidebar.number_input("VIX for Gold", value=16.0)
    inv = st.sidebar.number_input("Oil inventory change (%)", value=0.0)
    opec = st.sidebar.slider("OPEC quota adjustment", -1.0, 1.0, 0.0)
    pmi  = st.sidebar.number_input("Global PMI", value=50.0)
    return vix, inv, opec, pmi

# === Price Functions ===
def price_gold(rt_pct, vix, geo_score, eq_corr):
    return 2000*(1-rt_pct*0.1) + vix*10 + geo_score*300 - eq_corr*200

def price_oil(inv, opec, pmi, geo_score):
    p_term   = (pmi - 50)/100
    geo_term = (geo_score - 0.3)*100
    base     = 80*(1 + p_term - inv/100)
    return base + opec*80 + geo_term

def nelson_siegel_yield(short_rate_pct, tau=10, beta0=0.02, beta1=0.03, beta2=0.01, lam=0.6):
    # level component = short rate
    level = short_rate_pct/100
    # slope loading at tau=10
    slope = beta1 * (1 - np.exp(-lam*tau)) / (lam*tau)
    # curvature loading at tau=10
    curve = beta2 * ((1 - np.exp(-lam*tau)) / (lam*tau) - np.exp(-lam*tau))
    return level + slope + curve

# === Main Application ===
def run():
    eps, spx, a_gold, a_oil, a_10y, geo_ev = step1_market()
    liq, fiscal, geo, rt, m2 = step2_backdrop()
    regimes, probs = step3_regimes(liq, fiscal, geo)
    vals, rets, eps_ls, pe_ls, corrs = step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs)
    w_eps = sum(probs[r]/100 * eps_ls[i] for i,r in enumerate(regimes))
    w_pe  = sum(probs[r]/100 * pe_ls[i] for i,r in enumerate(regimes))
    fair_spx = w_eps * w_pe
    dfv = pd.DataFrame({'Regime':regimes, 'Fair SPX':vals, 'Return%':rets, 'P%':[probs[r] for r in regimes]})
    dfv['Fair SPX'] = dfv['Fair SPX'].apply(lambda x: f"${x:,.0f}")
    dfv['Return%']  = dfv['Return%'].apply(lambda x: f"{x:.1%}")
    dfv['P%']       = dfv['P%'].apply(lambda x: f"{x:.1f}%")
    st.subheader("Regime Fair-Value Table")
    st.table(dfv)
    vix, inv, opec, pmi = step6_anchors_inputs()
    avg_geo = np.mean(list(geo_ev.values()))
    avg_corr = np.mean(list(corrs.values()))
    gold_m = price_gold(rt, vix, avg_geo, avg_corr)
    oil_m  = price_oil(inv, opec, pmi, avg_geo)
    bond_y = nelson_siegel_yield(rt)
    anchors = pd.DataFrame(
        index=["SPX","Weighted EPS","Weighted P/E","Gold","Oil","10Y Yield"],
        data={
            "Actual": [spx, eps, spx/eps, a_gold, a_oil, a_10y/100],
            "Model":  [fair_spx, w_eps, w_pe, gold_m, oil_m, bond_y]
        }
    )
    fmt = anchors.copy()
    for m in fmt.index:
        for c in fmt.columns:
            v = anchors.loc[m,c]
            if m in ["SPX","Weighted EPS","Gold","Oil"]: fmt.loc[m,c] = f"${v:,.0f}"
            elif m=="Weighted P/E": fmt.loc[m,c] = f"{v:.1f}"
            else: fmt.loc[m,c] = f"{v:.1%}"
    st.subheader("Valuation & Asset Price Anchors")
    st.table(fmt)
    dfp, alloc, beta = portfolio_editor()
    exp_eq = sum(probs[r]/100 * rets[i] for i,r in enumerate(regimes)) * beta
    ret_asset = np.array([exp_eq, gold_m/a_gold-1, oil_m/a_oil-1, bond_y, DEFAULT_CASH_YIELD])
    exp_ret = alloc @ ret_asset
    st.subheader("Expected Portfolio Return")
    st.metric("Expected Return", f"{exp_ret:.2%}")
    vols = np.array([0.15,0.10,0.12,0.08,0.00])
    cov = np.diag(vols) @ np.array(
        [[1,avg_corr,0,0,0],[avg_corr,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
    ) @ np.diag(vols)
    sims = np.random.multivariate_normal(ret_asset, cov, 3000)
    st.subheader("Portfolio MC Distribution")
    st.line_chart(pd.Series(sims.mean(axis=1)).rolling(50).mean())

if __name__ == '__main__':
    run()
