# app.py — Streamlit Macro Scenario-Based Portfolio Simulator (Fixed and Explained)

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Macro Portfolio Simulator", layout="wide")

# === Constants ===
MAX_MACRO_PE_IMPACT = 0.6
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
    use_empirical = st.sidebar.checkbox("Use empirical macro inputs", value=True)

    if use_empirical:
        st.sidebar.markdown("#### Empirical Inputs")
        fed_balance_sheet = st.sidebar.number_input("Fed Balance Sheet (% of GDP)", value=35.0)
        short_term_rate = st.sidebar.number_input("Real Short-Term Rate (%)", value=1.5)
        m2_growth = st.sidebar.number_input("M2 Growth YoY (%)", value=4.0)

        deficit = st.sidebar.number_input("Federal Budget Deficit (% of GDP)", value=6.0)
        gov_spending = st.sidebar.number_input("Gov. Spending (% of GDP)", value=25.0)
        transfer_payments = st.sidebar.number_input("Net Transfers (% of GDP)", value=10.0)

        geo_risk_index = st.sidebar.number_input("Geopolitical Risk Index", value=120.0)
        vix_index = st.sidebar.number_input("VIX Volatility Index", value=20.0)
        conflict_events = st.sidebar.number_input("Global Conflict Events (count)", value=30)

        # Normalize based on assumed historical ranges
        liquidity_components = [
            (fed_balance_sheet - 15) / (45 - 15),
            max(min((5 - short_term_rate) / 5, 1), 0),  # inverse of rate
            max(min((m2_growth - 0) / (15 - 0), 1), 0)
        ]
        liq = sum(liquidity_components) / len(liquidity_components)

        fiscal_components = [
            (deficit - 1) / (15 - 1),
            (gov_spending - 15) / (35 - 15),
            (transfer_payments - 5) / (20 - 5)
        ]
        fiscal = sum([min(max(c, 0), 1) for c in fiscal_components]) / len(fiscal_components)

        geo_components = [
            (geo_risk_index - 50) / (200 - 50),
            (vix_index - 10) / (50 - 10),
            (conflict_events - 10) / (60 - 10)
        ]
        geo = sum([min(max(c, 0), 1) for c in geo_components]) / len(geo_components)

        st.sidebar.markdown("#### Derived Backdrop (0 to 1 scale)")
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, liq, disabled=False)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, fiscal, disabled=False)
        geo = st.sidebar.slider("Geopolitical Risk", 0.0, 1.0, geo, disabled=False)
    else:
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geopolitical Risk", 0.0, 1.0, 0.1)

    return liq, fiscal, geo

def scenario_probabilities(auto, liq, fiscal, geo):
    st.sidebar.header("Scenario Probabilities")
    names = ["Recession", "Stagflation", "Boom", "Deflation"]

    if auto:
        score = liq * 0.4 + fiscal * 0.3 - geo * 0.3
        probs = {
            "Boom": max(min(int(30 + score * 100), 60), 5),
            "Stagflation": max(min(int(25 + geo * 30 - liq * 10), 45), 5),
            "Recession": max(min(int(25 + (0.5 - liq) * 40), 50), 5),
            "Deflation": 100  # placeholder, will be balanced below
        }
        remaining = 100 - sum([v for k, v in probs.items() if k != "Deflation"])
        probs["Deflation"] = max(min(remaining, 40), 0)

            
        for s in names:
            st.sidebar.number_input(f"{s} % (auto)", 0, 100, probs[s], key=f"auto_{s}", disabled=True)
        return probs, True

    else:
        probs = {}
        total = 0
        for s in names:
            p = st.sidebar.number_input(f"{s} %", 0, 100, 25, key=f"manual_{s}")
            probs[s] = p
            total += p
        return (probs, total == 100)

def default_scenarios():
    return {
        "Recession": {"pe": 14, "eps_change": -0.20},
        "Stagflation": {"pe": 15, "eps_change": -0.10},
        "Boom": {"pe": 24, "eps_change": 0.15},
        "Deflation": {"pe": 17, "eps_change": -0.05},
    }

def macro_multiplier(liq, fiscal, geo):
    # Edge-case aware macro multiplier
    liquidity_impact = liq * 0.25  # larger upward push when fully flooded
    fiscal_impact = fiscal * 0.2   # stronger multiplier when stimulus is high
    geo_impact = -geo * 0.3        # more negative impact if geopolitical risk is high
    combined = liquidity_impact + fiscal_impact + geo_impact
    return 1 + min(MAX_MACRO_PE_IMPACT, combined)

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
                r = 0  # No change in bond yield, assuming flat environment; update if dynamic yield logic added
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
    trailing_pe = spx / eps if eps != 0 else 0
    st.markdown(f"**1. Trailing P/E (SPX / EPS):** {spx:,.0f} / {eps:.2f} = {trailing_pe:.2f}")
    st.markdown(f"**2. Weighted Forward EPS:** {weighted_eps:.2f} (based on scenario-weighted earnings)")
    st.markdown(f"**3. Weighted P/E:** {weighted_pe:.2f} (based on scenario-weighted multiples)")
    st.markdown(f"**4. Macro Multiplier:** {macro_mult:.3f} (adjusting for liquidity, stimulus, and risk)")
    st.markdown(f"**5. Fair SPX Estimate:** {weighted_eps:.2f} × {weighted_pe:.2f} × {macro_mult:.3f} = {fair_spx:,.0f}")

    st.subheader("Macro-Adjusted Asset Targets")
    st.markdown(f"- **Gold Target Price:** ${targets['Gold']:.2f}")
    st.markdown(f"- **Crude Oil Target Price:** ${targets['Crude']:.2f}")
    st.markdown(f"- **10-Year Yield Estimate:** {targets['10Y']:.2f}%")

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
