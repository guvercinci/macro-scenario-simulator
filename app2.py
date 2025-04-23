# app.py — Streamlit Macro Scenario-Based Portfolio Simulator (Fixed and Explained)

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Macro Portfolio Simulator", layout="wide")

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.3
DEFAULT_CASH_YIELD = 0.04
BOND_DURATION = 7

# === Input Functions ===
def valuation_inputs():
    st.sidebar.header("Market Inputs")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", value=200.0)
    spx = st.sidebar.number_input("Current SPX Index", value=5300.0)
    return eps, spx

def macro_conditions():
    st.sidebar.header("Macro Backdrop")
    liq = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
    fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
    geo = st.sidebar.slider("Geopolitical Risk", 0.0, 1.0, 0.1)
    return liq, fiscal, geo

def scenario_probabilities(auto, liq, fiscal, geo):
    st.sidebar.header("Scenario Probabilities")
    names = ["Recession", "Stagflation", "Boom", "Deflation"]
    if auto:
        if liq > 0.6 and fiscal > 0.5 and geo < 0.3:
            return {"Boom": 50, "Stagflation": 15, "Recession": 15, "Deflation": 20}, True
        elif liq < 0.3 and fiscal < 0.3:
            return {"Boom": 5, "Stagflation": 30, "Recession": 40, "Deflation": 25}, True
        elif geo > 0.6:
            return {"Boom": 10, "Stagflation": 35, "Recession": 35, "Deflation": 20}, True
        else:
            return {s: 25 for s in names}, True
    else:
        probs = {}
        total = 0
        for s in names:
            p = st.sidebar.number_input(f"{s} %", 0, 100, 25)
            probs[s] = p
            total += p
        return (probs, total == 100)

def default_scenarios():
    return {
        "Recession": {"pe": 14, "eps_change": -0.20},
        "Stagflation": {"pe": 15, "eps_change": -0.10},
        "Boom": {"pe": 20, "eps_change": 0.15},
        "Deflation": {"pe": 17, "eps_change": -0.05},
    }

def macro_multiplier(liq, fiscal, geo):
    return 1 + min(MAX_MACRO_PE_IMPACT, liq * 0.10 + fiscal * 0.05 - geo * 0.07)

def macro_targets(liq, fiscal, geo):
    return {
        "Gold": 2000 * (1 + liq * 0.05 + geo * 0.1),
        "Crude": 80 * (1 + fiscal * 0.1),
        "10Y": 4.0 * (1 - liq * 0.05 - geo * 0.05 + fiscal * 0.05),
    }

def portfolio_editor():
    st.subheader("Portfolio Allocation")
    return st.data_editor(pd.DataFrame({
        "symbol": ["Stocks", "Treasuries", "Commodities", "Cash", "Gold", "SPY Put Spread"],
        "allocation": [250000, 30000, 20000, 50000, 10000, 10000]
    }), use_container_width=True)

def simulate(eps, spx, probs, scenarios, macro_mult, alloc, targets):
    total = alloc["allocation"].sum()
    weighted_eps = sum(probs[s]/100 * eps * (1 + scenarios[s]['eps_change']) for s in probs)
    weighted_pe = sum(probs[s]/100 * scenarios[s]['pe'] for s in probs)
    fair_spx = weighted_eps * weighted_pe * macro_mult

    alloc["expected_return"] = 0.0
    for s, weight in probs.items():
        implied_spx = eps * (1 + scenarios[s]['eps_change']) * scenarios[s]['pe'] * macro_mult
        for i, row in alloc.iterrows():
            asset = row["symbol"]
            if asset == "Stocks":
                r = (implied_spx / spx) - 1
            elif asset == "Treasuries":
                r = BOND_DURATION * (targets["10Y"] - targets["10Y"]) / 100
            elif asset == "Commodities":
                r = 0.5 * ((targets["Crude"] / 80 - 1) + (targets["Gold"] / 2000 - 1))
            elif asset == "Gold":
                r = (targets["Gold"] / 2000) - 1
            elif asset == "Cash":
                r = DEFAULT_CASH_YIELD
            elif asset == "SPY Put Spread":
                fall = (spx - implied_spx) / spx if spx != 0 else 0
                r = min(max(fall, 0), 0.2) * 3
            else:
                r = 0
            alloc.at[i, "expected_return"] += (weight / 100) * r

    alloc["expected_dollar_return"] = alloc["allocation"] * alloc["expected_return"]
    alloc["final_value"] = alloc["allocation"] + alloc["expected_dollar_return"]
    return alloc, fair_spx, weighted_eps, weighted_pe

def main():
    st.title("Macro Scenario-Based Portfolio Simulator")
    st.markdown("""
This tool helps simulate how a diversified portfolio might respond across different macroeconomic scenarios.
We combine scenario-weighted earnings (EPS) and valuation (P/E ratio), then apply macro adjustments to estimate SPX fair value.
Asset class returns are then computed based on these conditions.
""")

    eps, spx = valuation_inputs()
    liq, fiscal, geo = macro_conditions()
    auto = st.sidebar.checkbox("Auto-adjust scenario probabilities", value=True)
    probs, valid = scenario_probabilities(auto, liq, fiscal, geo)
    if not valid:
        st.warning("Scenario probabilities must sum to 100%.")
        st.stop()

    scenarios = default_scenarios()
    macro_mult = macro_multiplier(liq, fiscal, geo)
    targets = macro_targets(liq, fiscal, geo)
    alloc = portfolio_editor()
    results, fair_spx, weighted_eps, weighted_pe = simulate(eps, spx, probs, scenarios, macro_mult, alloc, targets)

    st.subheader("Calculation Summary")
    st.markdown(f"**Weighted EPS:** {weighted_eps:.2f}")
    st.markdown(f"**Weighted P/E Ratio:** {weighted_pe:.2f}")
    st.markdown(f"**Macro Multiplier Applied:** {macro_mult:.3f}")
    st.markdown(f"**Fair SPX Estimate:** {fair_spx:,.0f}")

    st.subheader("Simulation Results")
    st.dataframe(results.style.format({
        "allocation": "$ {:,.0f}",
        "expected_dollar_return": "$ {:,.0f}",
        "final_value": "$ {:,.0f}",
        "expected_return": "{:.2%}"
    }))
    st.metric("Fair SPX Estimate", f"{fair_spx:,.0f}")
    st.metric("Expected Portfolio Return", f"{results['expected_dollar_return'].sum() / results['allocation'].sum():.2%}")
    st.metric("Expected Final Value", f"$ {results['final_value'].sum():,.0f}")

if __name__ == "__main__":
    main()
