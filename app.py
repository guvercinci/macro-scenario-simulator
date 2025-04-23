# Macro Scenario-Based Portfolio Simulator with Web UI (Streamlit)

import streamlit as st
import pandas as pd

st.title("Macro Scenario-Based Portfolio Simulator")
st.markdown("""
This tool simulates portfolio returns based on macroeconomic scenarios.
It uses assumptions about SPX valuation, earnings sensitivity, commodity prices,
interest rate shifts, and macro factors inspired by Ray Dalio's framework.

**How macro factors influence assets:**
- **Liquidity**: High liquidity boosts equity valuations and gold prices, and depresses yields.
- **Fiscal Stimulus**: Increases earnings expectations, equity multiples, and demand for energy.
- **Geopolitical Risk**: Suppresses equity P/E ratios, boosts gold (safe haven), and lowers bond yields (flight to safety).
""")

# Editable allocation chart
st.header("Adjust Portfolio Allocations ($)")
alloc_df = pd.DataFrame(
    {
        "symbol": ["Stocks", "Treasuries", "Commodities", "Cash", "Gold", "SPY Put Spread"],
        "allocation": [234434, 30000, 20000, 47000, 10000, 9260]
    }
)
editable_allocations = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

# Market valuation
st.header("Market Valuation Assumptions")
col1, col2 = st.columns(2)
with col1:
    trailing_eps = st.number_input("Trailing SPX Earnings (EPS)", value=200.27, step=1.0)
with col2:
    current_spx = st.number_input("Current SPX Index Level", value=5287.76, step=10.0)

trailing_pe = current_spx / trailing_eps
st.markdown(f"**Calculated Trailing P/E Ratio:** {trailing_pe:.2f}")

# Global macro backdrop
st.header("Global Macro Backdrop")
col3, col4, col5 = st.columns(3)
with col3:
    liquidity_index = st.slider("Liquidity Index (0=Drained, 1=Flooded)", 0.0, 1.0, 0.5)
with col4:
    fiscal_stimulus = st.slider("Fiscal Stimulus (0=None, 1=Extreme)", 0.0, 1.0, 0.3)
with col5:
    geopolitical_risk = st.slider("Geopolitical Risk Index (0=Stable, 1=Crisis)", 0.0, 1.0, 0.1)

# Macro-driven dynamic variables
adjusted_gold = 2000 * (1 + liquidity_index * 0.05 + geopolitical_risk * 0.1)
adjusted_crude = 80 * (1 + fiscal_stimulus * 0.1)
adjusted_10y = 4.0 * (1 - liquidity_index * 0.05 - geopolitical_risk * 0.05 + fiscal_stimulus * 0.05)

# Display adjusted variables
st.subheader("Macro-Adjusted Asset Anchors")
st.write(f"**Gold Price:** ${adjusted_gold:.2f}")
st.write(f"**Crude Oil Price:** ${adjusted_crude:.2f}")
st.write(f"**10-Year Yield:** {adjusted_10y:.2f}%")

reference_prices = {
    "SPX": current_spx,
    "Gold": adjusted_gold,
    "Crude": adjusted_crude,
    "10Y": adjusted_10y,
    "Trailing_EPS": trailing_eps,
    "Liquidity Index": liquidity_index,
    "Fiscal Stimulus": fiscal_stimulus,
    "Geopolitical Risk": geopolitical_risk
}

# Scenario assumptions
st.header("Scenario Assumptions")
default_scenario_data = {
    "Recession": {"pe": 14, "eps_change": -0.20},
    "Stagflation": {"pe": 15, "eps_change": -0.10},
    "Boom": {"pe": 20, "eps_change": 0.15},
    "Deflation": {"pe": 17, "eps_change": -0.05}
}

use_custom_assumptions = st.checkbox("Manually edit P/E and EPS change assumptions", value=False)
scenario_names = list(default_scenario_data.keys())
if use_custom_assumptions:
    assumption_input = pd.DataFrame(default_scenario_data).T.reset_index().rename(columns={"index": "Scenario", "pe": "P/E Ratio", "eps_change": "Earnings Change (%)"})
    assumption_input["Earnings Change (%)"] *= 100
    edited_assumptions = st.data_editor(assumption_input, num_rows="fixed", use_container_width=True)
    scenario_data = {
        row["Scenario"]: {"pe": row["P/E Ratio"], "eps_change": row["Earnings Change (%)"] / 100}
        for _, row in edited_assumptions.iterrows()
    }
else:
    scenario_data = default_scenario_data
    st.dataframe(pd.DataFrame(default_scenario_data).T.rename(columns={"pe": "P/E Ratio", "eps_change": "Earnings Change"}).style.format({"Earnings Change": "{:.0%}"}))

# Scenario probabilities
st.header("Scenario Probabilities")
probabilities = {}
total_prob = 0
for scenario in scenario_names:
    p = st.slider(f"{scenario} Probability (%)", min_value=0, max_value=100, value=25)
    probabilities[scenario] = p
    total_prob += p

st.write(f"**Total Assigned Probability:** {total_prob:.1f}%")
if total_prob != 100:
    st.warning("Total probabilities must sum to 100%.")

# Run simulation
if total_prob == 100:
    df = editable_allocations.copy()
    df["expected_return"] = 0.0

    for scenario, weight in probabilities.items():
        pe = scenario_data[scenario]["pe"]
        eps = reference_prices["Trailing_EPS"] * (1 + scenario_data[scenario]["eps_change"])
        implied_spx_price = eps * pe

        # Apply macro multipliers
        liquidity_boost = 1 + reference_prices["Liquidity Index"] * 0.1
        stimulus_boost = 1 + reference_prices["Fiscal Stimulus"] * 0.05
        geopolitical_drag = 1 - reference_prices["Geopolitical Risk"] * 0.05
        macro_multiplier = liquidity_boost * stimulus_boost * geopolitical_drag
        implied_spx_price *= macro_multiplier

        gold_target = reference_prices["Gold"]
        crude_price = reference_prices["Crude"]
        bond_yield = reference_prices["10Y"]

        for asset in df["symbol"]:
            if asset == "Stocks":
                r = (implied_spx_price / reference_prices["SPX"]) - 1
            elif asset == "Treasuries":
                r = (reference_prices["10Y"] - bond_yield) / reference_prices["10Y"]
            elif asset == "Commodities":
                crude_return = (crude_price / 80) - 1
                gold_return = (gold_target / 2000) - 1
                r = 0.5 * (crude_return + gold_return)
            elif asset == "Gold":
                r = (gold_target / 2000) - 1
            elif asset == "SPY Put Spread":
                r = (reference_prices["SPX"] - implied_spx_price) / reference_prices["SPX"] * 3
            else:
                r = 0

            df.loc[df["symbol"] == asset, "expected_return"] += (weight / 100) * r

    df["expected_dollar_return"] = df["allocation"] * df["expected_return"]
    df["final_value"] = df["allocation"] + df["expected_dollar_return"]

    st.subheader("Simulation Results")
    st.dataframe(df.style.format({"allocation": "$ {:,.0f}", "expected_dollar_return": "$ {:,.0f}", "final_value": "$ {:,.0f}", "expected_return": "{:.2%}"}))
    st.metric("Expected Portfolio Return", f"{df['expected_dollar_return'].sum() / df['allocation'].sum():.2%}")
    st.metric("Expected Final Portfolio Value", f"$ {df['final_value'].sum():,.0f}")
