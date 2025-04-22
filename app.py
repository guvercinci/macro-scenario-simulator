# Macro Scenario-Based Portfolio Simulator with Web UI (Streamlit)

import streamlit as st
import pandas as pd

st.title("Macro Scenario-Based Portfolio Simulator")

# Editable allocation chart
st.header("Adjust Portfolio Allocations ($)")
alloc_df = pd.DataFrame(
    {
        "symbol": ["Stocks", "Treasuries", "Commodities", "Cash", "Gold", "SPY Put Spread"],
        "allocation": [234434, 30000, 20000, 47000, 10000, 9260]
    }
)
editable_allocations = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

# Reference values
reference_prices = {
    "SPX": 4700,
    "Gold": 2000,
    "Crude": 80,
    "10Y": 4.0
}

# Assumed PE ratios for SPX per scenario
pe_ratios = {
    "Recession": 14,
    "Stagflation": 15,
    "Boom": 20,
    "Deflation": 17
}

# Assumed Gold prices per scenario
gold_targets = {
    "Recession": 2200,
    "Stagflation": 2500,
    "Boom": 1900,
    "Deflation": 2100
}

# Input: User's earnings expectation for SPX
st.header("Earnings Forecast")
earnings_input = st.number_input("Expected S&P 500 Index Earnings (EPS)", value=235)

# Scenario probabilities
st.header("Enter Scenario Probabilities")
scenario_names = ["Recession", "Stagflation", "Boom", "Deflation"]
probabilities = {}
total_prob = 0
for scenario in scenario_names:
    p = st.slider(f"{scenario} Probability (%)", min_value=0, max_value=100, value=25)
    probabilities[scenario] = p
    total_prob += p

st.write(f"**Total Assigned Probability:** {total_prob:.1f}%")

if total_prob != 100:
    st.warning("Total probabilities must sum to 100% to reflect reality.")

# Run simulation
if total_prob == 100:
    df = editable_allocations.copy()
    df["expected_return"] = 0.0

    for scenario, weight in probabilities.items():
        pe = pe_ratios[scenario]
        gold_target = gold_targets[scenario]
        implied_spx_price = earnings_input * pe
        crude_price = reference_prices["Crude"] * (0.9 if scenario == "Recession" else 1.2 if scenario == "Boom" else 1.0)
        bond_yield = reference_prices["10Y"] * (1.25 if scenario == "Boom" else 0.75 if scenario == "Recession" else 1.0)

        for asset in df["symbol"]:
            if asset == "Stocks":
                r = (implied_spx_price / reference_prices["SPX"]) - 1
            elif asset == "Treasuries":
                r = (reference_prices["10Y"] - bond_yield) / reference_prices["10Y"]
            elif asset == "Commodities":
                crude_return = (crude_price / reference_prices["Crude"]) - 1
                gold_return = (gold_target / reference_prices["Gold"]) - 1
                r = 0.5 * (crude_return + gold_return)
            elif asset == "Gold":
                r = (gold_target / reference_prices["Gold"]) - 1
            elif asset == "SPY Put Spread":
                r = (reference_prices["SPX"] - implied_spx_price) / reference_prices["SPX"] * 3
            else:
                r = 0

            df.loc[df["symbol"] == asset, "expected_return"] += (weight / 100) * r

    df["expected_dollar_return"] = df["allocation"] * df["expected_return"]
    df["final_value"] = df["allocation"] + df["expected_dollar_return"]

    portfolio_expected_return = df["expected_dollar_return"].sum() / df["allocation"].sum()
    portfolio_final_value = df["final_value"].sum()

    st.subheader("Simulation Results")
    st.dataframe(df.style.format({"allocation": "$ {:,.0f}", "expected_dollar_return": "$ {:,.0f}", "final_value": "$ {:,.0f}", "expected_return": "{:.2%}"}))
    st.metric("Expected Portfolio Return", f"{portfolio_expected_return:.2%}")
    st.metric("Expected Final Portfolio Value", f"$ {portfolio_final_value:,.0f}")

    st.subheader("Scenario Assumptions")
    assumption_df = pd.DataFrame({
        "Scenario": scenario_names,
        "P/E Ratio": [pe_ratios[s] for s in scenario_names],
        "SPX Price": [earnings_input * pe_ratios[s] for s in scenario_names],
        "Gold Price": [gold_targets[s] for s in scenario_names],
        "Crude Oil Price": [reference_prices["Crude"] * (0.9 if s == "Recession" else 1.2 if s == "Boom" else 1.0) for s in scenario_names],
        "10Y Yield": [reference_prices["10Y"] * (1.25 if s == "Boom" else 0.75 if s == "Recession" else 1.0) for s in scenario_names]
    })
    st.dataframe(assumption_df.set_index("Scenario"))
