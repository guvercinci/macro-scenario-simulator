# app.py — Macro Scenario Simulator: easy, guided, and transparent

import streamlit as st
import pandas as pd
import numpy as np

# === Page Config & Title ===
st.set_page_config(page_title="Macro Scenario Simulator", layout="wide")
st.title("Macro Scenario Simulator")
st.markdown(
    "This interactive tool helps you stress-test a portfolio across different economic regimes. "
    "Follow the numbered steps in the sidebar to set the macro backdrop, enter market prices, "
    "adjust scenario drivers, and visualize fair-value estimates and risk distributions."
)

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.02

# === Step 1: Macro Backdrop ===
LIQ_WEIGHTS = [0.4, 0.3, 0.3]
FISCAL_WEIGHTS = [0.33, 0.33, 0.34]
GEO_WEIGHTS = [0.5, 0.3, 0.2]

def normalize_liquidity(fed_bs, short_rate, m2):
    comps = [(fed_bs - 15) / 30, np.clip((5 - short_rate) / 5, 0, 1), np.clip(m2 / 15, 0, 1)]
    return sum(w * c for w, c in zip(LIQ_WEIGHTS, comps))

def normalize_fiscal(deficit, spend, transfer):
    comps = [np.clip((deficit - 1) / 14, 0, 1), np.clip((spend - 15) / 20, 0, 1), np.clip((transfer - 5) / 15, 0, 1)]
    return sum(w * c for w, c in zip(FISCAL_WEIGHTS, comps))

def normalize_geo(idx, vix, conflicts):
    comps = [np.clip((idx - 50) / 150, 0, 1), np.clip((vix - 10) / 40, 0, 1), np.clip((conflicts - 10) / 50, 0, 1)]
    return sum(w * c for w, c in zip(GEO_WEIGHTS, comps))

def step1_backdrop():
    st.sidebar.header("Step 1: Macro Backdrop")
    st.sidebar.markdown("*Empirical data or manual inputs for liquidity, fiscal, and geopolitical risk.*")
    use_emp = st.sidebar.checkbox("Use empirical inputs", True)
    if use_emp:
        override = st.sidebar.checkbox("Manual override", False)
        disabled = not override
        fed_bs = st.sidebar.number_input("Fed BS (% of GDP)", 38.0, disabled=disabled)
        real_rate = st.sidebar.number_input("Real short rate (%)", 2.5, disabled=disabled)
        m2 = st.sidebar.number_input("M2 growth YoY (%)", 6.0, disabled=disabled)
        deficit = st.sidebar.number_input("Budget deficit (% of GDP)", 5.0, disabled=disabled)
        spend = st.sidebar.number_input("Govt spending (% of GDP)", 24.0, disabled=disabled)
        transfer = st.sidebar.number_input("Net transfers (% of GDP)", 12.0, disabled=disabled)
        geo_idx = st.sidebar.number_input("Geo risk index", 100.0, disabled=disabled)
        vix = st.sidebar.number_input("VIX index", 16.0, disabled=disabled)
        conflicts = st.sidebar.number_input("Conflict events (#)", 20, disabled=disabled)
        liq = normalize_liquidity(fed_bs, real_rate, m2)
        fiscal = normalize_fiscal(deficit, spend, transfer)
        geo = normalize_geo(geo_idx, vix, conflicts)
        liq = st.sidebar.slider("Liquidity score", 0.0, 1.0, liq, disabled=not override)
        fiscal = st.sidebar.slider("Fiscal score", 0.0, 1.0, fiscal, disabled=not override)
        geo = st.sidebar.slider("Geo risk score", 0.0, 1.0, geo, disabled=not override)
    else:
        real_rate, m2 = 2.5, 6.0
        liq = st.sidebar.slider("Liquidity score", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal score", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geo risk score", 0.0, 1.0, 0.1)
    return liq, fiscal, geo, real_rate, m2

# === Step 2: Market Prices & Flashpoints ===
def step2_market():
    st.sidebar.header("Step 2: Market Prices & Flashpoints")
    st.sidebar.markdown("*Enter current prices and geo-event impacts.*")
    eps = st.sidebar.number_input("SPX trailing EPS", 220.0)
    spx = st.sidebar.number_input("SPX index level", 4500.0)
    st.sidebar.markdown("**Current Asset Prices**")
    a_gold = st.sidebar.number_input("Gold price ($)", 1900.0)
    a_oil = st.sidebar.number_input("Oil price ($)", 75.0)
    a_10y = st.sidebar.number_input("10Y yield (%)", 3.5)
    st.sidebar.markdown("**Geo flashpoints**")
    geo_events = {}
    for name in ["Tariff shock", "Gas cutoff"]:
        geo_events[name] = st.sidebar.slider(name, -1.0, 1.0, 0.0)
    return eps, spx, a_gold, a_oil, a_10y, geo_events

# === Step 3: Regime Probabilities ===
def step3_regimes(liq, fiscal, geo):
    st.sidebar.header("Step 3: Regime Probabilities")
    st.sidebar.markdown("*Auto-computed based on backdrop, then optional override.*")
    exp_p = max(liq * 0.6 + fiscal * 0.4 - geo * 0.2, 0)
    rec_p = max((1 - liq) * 0.7 + geo * 0.3, 0)
    stag_p = max(fiscal * 0.2 + geo * 0.5 + liq * 0.1, 0)
    defl_p = max((1 - fiscal) * 0.5 + (1 - geo) * 0.5 - liq * 0.2, 0)
    total = exp_p + rec_p + stag_p + defl_p
    auto = {'Expansion': exp_p / total, 'Recession': rec_p / total, 'Stagflation': stag_p / total, 'Deflation': defl_p / total}
    override = st.sidebar.checkbox("Override manually", False)
    probs = {}
    for r, pct in auto.items():
        default = int(pct * 100)
        probs[r] = st.sidebar.slider(f"P({r})%", 0, 100, default, disabled=not override)
    if not override:
        probs = {r: int(p * 100) for r, p in auto.items()}
    if override and sum(probs.values()) != 100:
        st.sidebar.error("Regime probabilities must sum to 100%.")
        st.stop()
    return list(auto.keys()), probs

# === Helper Functions ===
def pe_from_real(rate_pct, prem=0.04):
    real_rate = rate_pct / 100.0
    req = real_rate + prem
    pe = (1 / req) if req > 0 else float('inf')
    return min(40, max(8, pe))

def nelson_siegel(rt):
    return 1 - ((1 - np.exp(-rt)) / rt) + 0.5 * (((1 - np.exp(-rt)) / rt) - np.exp(-rt))

def price_gold(rt, vix, geo_score):
    return 2000 * (1 - rt * 0.1) + vix * 10 + geo_score * 300

def price_oil(inv, opec, pmi, geo_score):
    return 80 * (1 + pmi / 100 - inv / 100) + opec * 80 + geo_score * 50

def eps_proj(eps, gdp, inf, ratec, sharec):
    rev = eps * (1 + gdp / 100)
    marg = rev * (1 - inf * 0.005 - ratec * 0.01)
    debt = ratec * 0.1
    floor = eps * 0.125
    return max(marg - debt, floor) * (1 - sharec)

def macro_mult(liq, fiscal, geo):
    impact = liq * 0.25 + fiscal * 0.2 - geo * 0.3
    return 1 + min(MAX_MACRO_PE_IMPACT, impact)

def simulate(alloc, ret, cov, sims=3000):
    draws = np.random.multivariate_normal(ret, cov, sims)
    return (draws * alloc).sum(axis=1)

def portfolio_editor():
    st.subheader("Step 4: Portfolio Allocation")
    df_init = pd.DataFrame({'Asset': ['Equities', 'Gold', 'Oil', 'Bonds', 'Cash'], 'Pct': [40, 20, 20, 15, 5]})
    df = st.data_editor(df_init, use_container_width=True) if hasattr(st, 'data_editor') else st.experimental_data_editor(df_init, use_container_width=True)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Portfolio weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values / 100

# === Main Application ===
def run():
    # Steps 1-3
    liq, fiscal, geo, rt, m2 = step1_backdrop()
    eps, spx, a_gold, a_oil, a_10y, geo_events = step2_market()
    regimes, probs = step3_regimes(liq, fiscal, geo)

    # Step 5: Scenario drivers & correlations
    st.sidebar.header("Step 5: Scenario Drivers & Correlations")
    gdp_def = {'Expansion': 3.0, 'Recession': -1.0, 'Stagflation': 1.0, 'Deflation': -0.5}
    rate_def = {'Expansion': 0.2, 'Recession': 1.0, 'Stagflation': 0.8, 'Deflation': -0.2}
    share_def = {'Expansion': 0.0, 'Recession': 0.02, 'Stagflation': 0.0, 'Deflation': 0.0}
    values, rets, eps_list, pe_list, corr_vals = [], [], [], [], {}
    for reg in regimes:
        with st.sidebar.expander(reg, True):
            gdp = st.number_input(f"GDP {reg}%", gdp_def[reg], key=f"gdp_{reg}")
            ratec = st.number_input(f"Rate shock {reg}%", rate_def[reg], key=f"rs_{reg}")
            sharec = st.number_input(f"Share chg {reg}%", share_def[reg], key=f"sc_{reg}")
            corr_vals[reg] = st.slider(f"Eq-Gold corr {reg}", -1.0, 1.0, -0.2, key=f"corr_{reg}")
        eps_f = eps_proj(eps, gdp, m2, ratec, sharec)
        pe_f = pe_from_real(rt) * macro_mult(liq, fiscal, geo)
        fair_reg = eps_f * pe_f
        values.append(fair_reg)
        rets.append(fair_reg / spx - 1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)
    # Weighted EPS/P/E and fair SPX
    weighted_eps = sum(probs[r] / 100 * eps_list[i] for i, r in enumerate(regimes))
    weighted_pe = sum(probs[r] / 100 * pe_list[i] for i, r in enumerate(regimes))
    fair_spx = weighted_eps * weighted_pe
    # Display scenario table
    dfv = pd.DataFrame({'Regime': regimes, 'Fair SPX': values, 'Return%': rets, 'P%': [probs[r] for r in regimes]})
    st.write(dfv)

    # Step 6: Valuation & Asset Anchors
    st.subheader("Step 6: Valuation & Asset Price Anchors")
    gold_price = price_gold(rt, st.sidebar.number_input("VIX for Gold", 16.0), sum(geo_events.values()))
    oil_price = price_oil(
        st.sidebar.number_input("Oil inv change (%)", 0.0),
        st.sidebar.slider("OPEC quota", -1.0, 1.0, 0.0),
        st.sidebar.number_input("Global PMI", 50.0),
        sum(geo_events.values())
    )
    bond_yield = nelson_siegel(rt)
    anchors = pd.DataFrame(
        index=["SPX","Weighted EPS","Weighted P/E","Gold","Oil","10Y Yield"],
        data={
            "Actual": [spx, eps, spx/eps, a_gold, a_oil, a_10y/100],
            "Model": [fair_spx, weighted_eps, weighted_pe, gold_price, oil_price, bond_yield]
        }
    )
    st.table(anchors.style.format({"Actual": "{:.2f}", "Model": "{:.2f}"}))

    # Step 7: Portfolio & Expected Return
    dfp, alloc = portfolio_editor()
    exp_eq = sum(probs[r] / 100 * rets[i] for i, r in enumerate(regimes))
    ret_asset = np.array([exp_eq, gold_price / a_gold - 1, oil_price / a_oil - 1, bond_yield, DEFAULT_CASH_YIELD])
    exp_return = alloc @ ret_asset
    st.subheader("Step 7: Expected Portfolio Return")
    st.metric("Expected Return", f"{exp_return:.2%}")

    # Step 8: Monte Carlo Simulation & Distribution
    vols = np.array([0.15, 0.10, 0.12, 0.08, 0.00])
    avg_corr = np.mean(list(corr_vals.values()))
    cov = np.diag(vols) @ np.array([
        [1, avg_corr, 0, 0, 0],
        [avg_corr, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]) @ np.diag(vols)
    sims = simulate(alloc, ret_asset, cov)
    st.subheader("Step 8: Portfolio MC Distribution")
    st.line_chart(pd.Series(sims).rolling(50).mean())

if __name__ == '__main__':
    run()
