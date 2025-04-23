# app.py — Streamlit Macro Scenario-Based Portfolio Simulator (Fixed Version)

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Macro Portfolio Simulator", layout="wide")

MAX_MACRO_PE_IMPACT = 0.6
DEFAULT_CASH_YIELD = 0.04
BOND_DURATION = 7

def valuation_inputs():
    st.sidebar.header("Market Inputs")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", value=200.0)
    spx = st.sidebar.number_input("Current SPX Index", value=5300.0)
    return eps, spx

def macro_conditions():
    st.sidebar.header("Macro Backdrop")
    use_empirical = st.sidebar.checkbox("Use empirical macro inputs", value=True)
    allow_override = st.sidebar.checkbox("Override empirical backdrop manually", value=False)
    disabled_inputs = not allow_override

    # default values if not using empirical inputs
    short_term_rate = 1.5
    m2_growth = 4.0

    if use_empirical:
        st.sidebar.markdown("#### Empirical Inputs for Liquidity")
        fed_balance_sheet = st.sidebar.number_input(
            "Fed Balance Sheet (% of GDP)", value=35.0, disabled=disabled_inputs
        )
        short_term_rate = st.sidebar.number_input(
            "Real Short-Term Rate (%)", value=1.5, disabled=disabled_inputs
        )
        m2_growth = st.sidebar.number_input(
            "M2 Growth YoY (%)", value=4.0, disabled=disabled_inputs
        )

        st.sidebar.markdown("#### Empirical Inputs for Fiscal Stimulus")
        deficit = st.sidebar.number_input(
            "Federal Budget Deficit (% of GDP)", value=6.0, disabled=disabled_inputs
        )
        gov_spending = st.sidebar.number_input(
            "Gov. Spending (% of GDP)", value=25.0, disabled=disabled_inputs
        )
        transfer_payments = st.sidebar.number_input(
            "Net Transfers (% of GDP)", value=10.0, disabled=disabled_inputs
        )

        st.sidebar.markdown("#### Empirical Inputs for Geopolitical Risk")
        geo_risk_index = st.sidebar.number_input(
            "Geopolitical Risk Index", value=120.0, disabled=disabled_inputs
        )
        vix_index = st.sidebar.number_input(
            "VIX Volatility Index", value=20.0, disabled=disabled_inputs
        )
        conflict_events = st.sidebar.number_input(
            "Global Conflict Events (count)", value=30, disabled=disabled_inputs
        )

        # compute normalized components
        liquidity_components = [
            (fed_balance_sheet - 15) / (45 - 15),
            max(min((5 - short_term_rate) / 5, 1), 0),
            max(min((m2_growth - 0) / 15, 1), 0)
        ]
        liq = sum(liquidity_components) / len(liquidity_components)

        fiscal_components = [
            (deficit - 1) / (15 - 1),
            (gov_spending - 15) / (35 - 15),
            (transfer_payments - 5) / (20 - 5)
        ]
        fiscal = sum(min(max(c, 0), 1) for c in fiscal_components) / len(fiscal_components)

        geo_components = [
            (geo_risk_index - 50) / (200 - 50),
            (vix_index - 10) / (50 - 10),
            (conflict_events - 10) / (60 - 10)
        ]
        geo = sum(min(max(c, 0), 1) for c in geo_components) / len(geo_components)

        st.sidebar.markdown("#### Derived Macro Backdrop (0 to 1 scale)")
        liq = st.sidebar.slider(
            "Liquidity", 0.0, 1.0, liq, disabled=not allow_override
        )
        fiscal = st.sidebar.slider(
            "Fiscal Stimulus", 0.0, 1.0, fiscal, disabled=not allow_override
        )
        geo = st.sidebar.slider(
            "Geopolitical Risk", 0.0, 1.0, geo, disabled=not allow_override
        )
    else:
        liq = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
        fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
        geo = st.sidebar.slider("Geopolitical Risk", 0.0, 1.0, 0.1)

    return liq, fiscal, geo, short_term_rate, m2_growth

def pe_from_real_rate(real_rate, equity_risk_premium=0.04):
    required_return = real_rate + equity_risk_premium
    return min(40, max(8, 1 / required_return))

def default_scenarios(short_rate):
    return {
        "Recession":   {"pe": pe_from_real_rate(2.5), "eps_change": -0.20},
        "Stagflation": {"pe": pe_from_real_rate(2.8), "eps_change": -0.10},
        "Boom":        {"pe": pe_from_real_rate(1.0), "eps_change":  0.15},
        "Deflation":   {"pe": pe_from_real_rate(1.8), "eps_change": -0.05},
    }

def macro_multiplier(liq, fiscal, geo):
    liquidity_impact = liq * 0.25
    fiscal_impact = fiscal * 0.2
    geo_impact = -geo * 0.3
    return 1 + min(MAX_MACRO_PE_IMPACT, liquidity_impact + fiscal_impact + geo_impact)

def macro_targets(liq, fiscal, geo, short_term_rate=1.5, m2_growth=4.0):
    gold_base = 2000
    real_rate_factor = max(0, (3 - short_term_rate)) * 0.25
    m2_factor = max(0, (m2_growth - 5)) * 0.10
    geo_factor = geo * 0.4
    gold_price = gold_base * (1 + real_rate_factor + m2_factor + geo_factor)

    crude_price = 80 * (1 + fiscal * 0.2 + geo * 0.3 + liq * 0.1)
    bond_yield = 4.0 * (1 - liq * 0.05 - geo * 0.05 + fiscal * 0.05)

    return {
        "Gold": gold_price,
        "Crude": crude_price,
        "10Y": bond_yield
    }

def scenario_probabilities():
    st.sidebar.header("Scenario Probabilities")
    probs = {}
    total = 0
    for scenario in ["Recession", "Stagflation", "Boom", "Deflation"]:
        val = st.sidebar.slider(f"{scenario} Probability (%)", 0, 100, 25)
        probs[scenario] = val
        total += val
    st.sidebar.markdown(f"**Total Probability: {total}%**")
    if total != 100:
        st.sidebar.error("Total must equal 100%")
        st.stop()
    return probs

def portfolio_editor():
    st.subheader("Portfolio Allocation")
    portfolio_size = st.number_input("Total Portfolio Size ($)", value=100000, step=10000)
    equity_beta = st.sidebar.slider("Equity Beta (vs SPX)", 0.5, 2.0, 1.0, step=0.1)
    preset = st.selectbox("Portfolio Preset", ["Balanced", "Aggressive", "Defensive", "Custom"], index=0)

    presets = {
        "Balanced":   [60, 30, 5, 3, 2, 0],
        "Aggressive": [80, 10, 2, 5, 2, 1],
        "Defensive":  [30, 30, 10, 10, 10, 10],
        "Custom":     [50, 30, 10, 5, 3, 2]
    }
    names = ["Equities", "Fixed Income", "Cash", "Commodities", "Gold", "Hedging Instruments"]
    df = pd.DataFrame({"symbol": names, "allocation_pct": presets[preset]})
    df = st.data_editor(df, num_rows="fixed", use_container_width=True)
    total_pct = df["allocation_pct"].sum()
    st.markdown(f"**Total Allocation: {total_pct:.1f}%**")
    if abs(total_pct - 100) > 0.1:
        st.error("Total allocation must equal 100%.")
        st.stop()
    df["allocation"] = df["allocation_pct"] / 100 * portfolio_size
    df["beta"] = df["symbol"].apply(lambda x: equity_beta if x == "Equities" else 1.0)
    return df

def calculate_fair_value(eps, scenarios, probs, liq, fiscal, geo):
    weighted_eps = sum(probs[k] / 100 * eps * (1 + scenarios[k]["eps_change"]) for k in probs)
    weighted_pe = sum(probs[k] / 100 * scenarios[k]["pe"] for k in probs)
    macro_mult = macro_multiplier(liq, fiscal, geo)
    adjusted_pe = weighted_pe * macro_mult
    fair_spx = weighted_eps * adjusted_pe
    return weighted_eps, weighted_pe, macro_mult, fair_spx

def run():
    st.title("Macro Scenario-Based Portfolio Simulator")
    eps, spx = valuation_inputs()
    liq, fiscal, geo, short_rate, m2 = macro_conditions()
    targets = macro_targets(liq, fiscal, geo, short_rate, m2)
    scenarios = default_scenarios(short_rate)
    probs = scenario_probabilities()
    weighted_eps, weighted_pe, macro_mult, fair_spx = calculate_fair_value(
        eps, scenarios, probs, liq, fiscal, geo
    )

    st.subheader("Calculation Summary")
    trailing_pe = spx / eps
    st.markdown(f"**1. Trailing P/E:** {spx:.0f} / {eps:.2f} = {trailing_pe:.2f}")
    st.markdown("_Shows current valuation of SPX vs earnings._")
    st.markdown(f"**2. Weighted Forward EPS:** {weighted_eps:.2f}")
    st.markdown("_This forecasts expected earnings based on how likely each scenario is to occur._")
    st.markdown(f"**3. Weighted P/E:** {weighted_pe:.2f}")
    st.markdown("_Average valuation across all scenarios._")
    st.markdown(f"**4. Macro Multiplier:** {macro_mult:.3f}")
    st.markdown("_Adjustment for liquidity, stimulus, and risk._")
    st.markdown(f"**5. Fair SPX Estimate:** {weighted_eps:.2f} × {weighted_pe:.2f} × {macro_mult:.3f} = {fair_spx:,.0f}")
    st.markdown("_Macro-adjusted valuation estimate._")

    st.subheader("Macro-Adjusted Asset Anchors")
    st.markdown(f"- **Gold:** ${targets['Gold']:.2f}")
    st.markdown(f"- **Crude Oil:** ${targets['Crude']:.2f}")
    st.markdown(f"- **10-Year Yield:** {targets['10Y']:.2f}%")

    df = portfolio_editor()
    st.subheader("Expected Portfolio Return")
    results = []
    for _, row in df.iterrows():
        sym, alloc, beta = row["symbol"], row["allocation"], row["beta"]
        if sym == "Equities":
            r = ((fair_spx / spx) - 1) * beta
        elif sym == "Fixed Income":
            r = BOND_DURATION * (targets["10Y"] - 4.0) / 100
        elif sym == "Cash":
            r = DEFAULT_CASH_YIELD
        elif sym == "Commodities":
            r = ((targets["Crude"] / 80) - 1)
        elif sym == "Gold":
            r = ((targets["Gold"] / 2000) - 1)
        elif sym == "Hedging Instruments":
            fall = max(0, 1 - (fair_spx / spx))
            r = min(max(fall * 3, -0.1), 0.3)
        else:
            r = 0
        results.append((sym, alloc, r, alloc * r))

    result_df = pd.DataFrame(results, columns=["Asset", "Allocation", "Return", "Gain/Loss"])
    st.dataframe(
        result_df.style.format({"Allocation": "$ {:,.0f}", "Return": "{:.2%}", "Gain/Loss": "$ {:,.0f}"})
    )
    st.metric(
        "Total Expected Return", f"{result_df['Gain/Loss'].sum() / result_df['Allocation'].sum():.2%}"
    )
    st.metric(
        "Final Portfolio Value", f"$ {result_df['Allocation'].sum() + result_df['Gain/Loss'].sum():,.0f}"
    )

run()
