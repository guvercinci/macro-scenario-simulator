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
MAX_MACRO_PE_IMACT = 0.6
DEFAULT_CASH_YIELD = 0.02

# === Market Prices & Flashpoints ===
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
    geo_events = {}
    for name in ["Tariff shock", "Gas cutoff"]:
        geo_events[name] = st.sidebar.slider(f"Impact: {name}", -1.0, 1.0, 0.0)
    return eps, spx, a_gold, a_oil, a_10y, geo_events

# === Macro Backdrop ===
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

# === Regime Probabilities ===
def step3_regimes(liq, fiscal, geo):
    st.sidebar.header("Regime Probabilities")
    st.sidebar.markdown("*Auto-computed from backdrop, optional override.*")
    # Raw score calculations
    exp_p = max(liq * 0.6 + fiscal * 0.4 - geo * 0.2, 0)
    rec_p = max((1 - liq) * 0.7 + geo * 0.3, 0)
    stag_p = max(fiscal * 0.2 + geo * 0.5 + liq * 0.1, 0)
    defl_p = max((1 - fiscal) * 0.5 + (1 - geo) * 0.5 - liq * 0.2, 0)
    total_score = exp_p + rec_p + stag_p + defl_p
    auto = {'Expansion': exp_p / total_score, 'Recession': rec_p / total_score,
            'Stagflation': stag_p / total_score, 'Deflation': defl_p / total_score}
    override = st.sidebar.checkbox("Override probabilities", False)
    probs = {}
    # Always show inputs (disabled if not override)
    for r, pct in auto.items():
        default = int(pct * 100)
        probs[r] = st.sidebar.number_input(
            f"P({r})%", min_value=0, max_value=100, value=default,
            disabled=not override, key=f"prob_{r}"
        )
    # Show total always
    total = sum(probs.values())
    st.sidebar.markdown(f"**Total Probability: {total}%**")
    # If override, enforce sum=100
    if override and total != 100:
        st.sidebar.error("Probabilities must sum to 100% when overriding.")
        st.stop()
    # If not overriding, normalize defaults to sum 100
    if not override:
        defaults = {r: int(auto[r] * 100) for r in auto}
        diff = 100 - sum(defaults.values())
        if diff != 0:
            key_max = max(defaults, key=defaults.get)
            defaults[key_max] += diff
        probs = defaults
    return list(auto.keys()), probs

# === Portfolio Allocation ===
def portfolio_editor():
    st.subheader("Portfolio Allocation")
    df_init = pd.DataFrame({'Asset': ['Equities', 'Gold', 'Oil', 'Bonds', 'Cash'], 'Pct': [40, 20, 20, 15, 5]})
    if hasattr(st, 'data_editor'):
        df = st.data_editor(df_init, use_container_width=True)
    else:
        df = st.experimental_data_editor(df_init, use_container_width=True)
    if abs(df['Pct'].sum() - 100) > 0.1:
        st.error("Weights must sum to 100%.")
        st.stop()
    return df, df['Pct'].values / 100

# === Scenario Drivers & Correlations ===
def step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs):
    st.sidebar.header("Scenario Drivers & Correlations")
    gdp_def = {'Expansion': 3.0, 'Recession': -1.0, 'Stagflation': 1.0, 'Deflation': -0.5}
    rate_def = {'Expansion': 0.2, 'Recession': 1.0, 'Stagflation': 0.8, 'Deflation': -0.2}
    share_def = {'Expansion': 0.0, 'Recession': 0.02, 'Stagflation': 0.0, 'Deflation': 0.0}
    values, rets, eps_list, pe_list, corr_vals = [], [], [], [], {}
    for reg in regimes:
        with st.sidebar.expander(reg, True):
            gdp = st.number_input(f"GDP {reg}%", gdp_def[reg], key=f"gdp_{reg}")
            ratec = st.number_input(f"Rate shock {reg}%", rate_def[reg], key=f"rs_{reg}")
            sharec = st.number_input(f"Share change {reg}%", share_def[reg], key=f"sc_{reg}")
            corr_vals[reg] = st.sidebar.slider(f"Eq-Gold correlation {reg}", -1.0, 1.0, -0.2, key=f"corr_{reg}")
        eps_f = eps_proj(eps, gdp, m2, ratec, sharec)
        pe_f = pe_from_real(rt) * macro_mult(liq, fiscal, geo)
        fair_reg = eps_f * pe_f
        values.append(fair_reg)
        rets.append(fair_reg / spx - 1)
        eps_list.append(eps_f)
        pe_list.append(pe_f)
    return values, rets, eps_list, pe_list, corr_vals

# === Anchor Drivers & Assumptions ===
def step6_anchors_inputs():
    st.sidebar.header("Anchor Drivers & Assumptions")
    st.sidebar.markdown("*Inputs for valuation and asset price anchors.*")
    vix_model = st.sidebar.number_input("VIX for Gold", value=16.0)
    inv_change = st.sidebar.number_input("Oil inventory change (%)", value=0.0)
    opec_quota = st.sidebar.slider("OPEC quota adjustment", -1.0, 1.0, 0.0)
    pmi_model = st.sidebar.number_input("Global PMI", value=50.0)
    return vix_model, inv_change, opec_quota, pmi_model

# === Helper Functions ===
def pe_from_real(rate_pct, prem=0.04):
    real_rate = rate_pct / 100.0
    req = real_rate + prem
    pe = 1 / req if req > 0 else float('inf')
    return min(40, max(8, pe))
def nelson_siegel(rt): return 1 - ((1 - np.exp(-rt)) / rt) + 0.5 * (((1 - np.exp(-rt)) / rt) - np.exp(-rt))

def price_gold(rt, vix, geo_score, eq_gold_corr):
    base = 2000 * (1 - rt * 0.1)
    vix_term = vix * 10
    geo_term = geo_score * 300
    corr_term = -eq_gold_corr * 200
    return base + vix_term + geo_term + corr_term

def price_oil(inv, opec, pmi, geo_score): return 80 * (1 + pmi / 100 - inv / 100) + opec * 80 + geo_score * 50
def eps_proj(eps, gdp, inf, ratec, sharec):
    rev = eps * (1 + gdp / 100)
    marg = rev * (1 - inf * 0.005 - ratec * 0.01)
    debt = ratec * 0.1
    floor = eps * 0.125
    return max(marg - debt, floor) * (1 - sharec)

def macro_mult(liq, fiscal, geo): return 1 + min(MAX_MACRO_PE_IMACT, liq * 0.25 + fiscal * 0.2 - geo * 0.3)
def simulate(alloc, ret, cov, sims=3000): draws = np.random.multivariate_normal(ret, cov, sims); return (draws * alloc).sum(axis=1)

# === Main Application ===
def run():
    eps, spx, a_gold, a_oil, a_10y, geo_events = step1_market()
    liq, fiscal, geo, rt, m2 = step2_backdrop()
    global regimes
    regimes, probs = step3_regimes(liq, fiscal, geo)
    values, rets, eps_list, pe_list, corr_vals = step5_drivers(eps, spx, rt, m2, liq, fiscal, geo, regimes, probs)
    weighted_eps = sum(probs[r] / 100 * eps_list[i] for i, r in enumerate(regimes))
    weighted_pe = sum(probs[r] / 100 * pe_list[i] for i, r in enumerate(regimes))
    fair_spx = weighted_eps * weighted_pe
    dfv = pd.DataFrame({'Regime': regimes, 'Fair SPX': values, 'Return%': rets, 'P%': [probs[r] for r in regimes]})
    # Format dfv: Fair SPX as $ no decimals, Return% and P% as percentages with one decimal
    fmt_dfv = dfv.copy()
    fmt_dfv['Fair SPX'] = fmt_dfv['Fair SPX'].apply(lambda x: f'${x:,.0f}')
    fmt_dfv['Return%'] = fmt_dfv['Return%'].apply(lambda x: f'{x:.1%}')
    fmt_dfv['P%'] = fmt_dfv['P%'].apply(lambda x: f'{x:.1f}%')
    st.subheader("Regime Fair-Value Table")
    st.table(fmt_dfv)
    vix_model, inv_change, opec_quota, pmi_model = step6_anchors_inputs()
    avg_corr = sum((probs[r] / 100) * corr_vals[r] for r in regimes)
    gold_price = price_gold(rt, vix_model, sum(geo_events.values()), avg_corr)
    oil_price = price_oil(inv_change, opec_quota, pmi_model, sum(geo_events.values()))
    bond_yield = nelson_siegel(rt)
        # Valuation & Asset Price Anchors
    anchors = pd.DataFrame(
        index=["SPX", "Weighted EPS", "Weighted P/E", "Gold", "Oil", "10Y Yield"],
        data={
            "Actual": [spx, eps, spx/eps, a_gold, a_oil, a_10y/100],
            "Model": [fair_spx, weighted_eps, weighted_pe, gold_price, oil_price, bond_yield]
        }
    )
    # Format currency and percentage displays
    fmt_anchors = anchors.copy()
    for metric in fmt_anchors.index:
        for col in fmt_anchors.columns:
            val = anchors.loc[metric, col]
            if metric in ["SPX", "Weighted EPS", "Gold", "Oil"]:
                fmt_anchors.loc[metric, col] = f"${val:,.0f}"
            elif metric == "Weighted P/E":
                fmt_anchors.loc[metric, col] = f"{val:.1f}"
            elif metric == "10Y Yield":
                fmt_anchors.loc[metric, col] = f"{val:.1%}"
    st.subheader("Valuation & Asset Price Anchors")
    st.table(fmt_anchors)
    # Portfolio Allocation
    dfp, alloc = portfolio_editor()
    exp_eq = sum(probs[r] / 100 * rets[i] for i, r in enumerate(regimes))
    ret_asset = np.array([exp_eq, gold_price / a_gold - 1, oil_price / a_oil - 1, bond_yield, DEFAULT_CASH_YIELD])
    exp_return = alloc @ ret_asset
    st.subheader("Expected Portfolio Return")
    st.metric("Expected Return", f"{exp_return:.2%}")
    vols = np.array([0.15, 0.10, 0.12, 0.08, 0.00])
    cov = np.diag(vols) @ np.array([
        [1, avg_corr, 0, 0, 0],
        [avg_corr, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]) @ np.diag(vols)
    sims = simulate(alloc, ret_asset, cov)
    st.subheader("Portfolio MC Distribution")
    st.line_chart(pd.Series(sims).rolling(50).mean())

if __name__ == '__main__':
    run()
