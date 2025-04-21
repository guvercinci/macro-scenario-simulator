# Macro Scenario-Based Portfolio Simulator with Web UI (Streamlit)

import streamlit as st
import pandas as pd

# Define default allocations
allocations = [
    {"symbol": "Stocks", "allocation": 234434},
    {"symbol": "Treasuries", "allocation": 30000},
    {"symbol": "Commodities", "allocation": 20000},
    {"symbol": "Cash", "allocation": 47000},
    {"symbol": "Gold", "allocation": 10000},
    {"symbol": "SPY Put Spread", "allocation": 9260}
]

# Define macro scenarios with default returns (in %)
scenarios = {
    "Recession": {
        "Stocks": -15,
        "Treasuries": 5,
        "Commodities": -5,
        "Cash": 0,
        "Gold": 3,
        "SPY Put Spread": 207
    },
    "Stagflation": {
        "Stocks": -10,
        "Treasuries": -8,
        "Commodities": 6,
        "Cash": 0,
        "Gold": 10,
        "SPY Put Spread": 100
    },
    "Boom": {
        "Stocks": 12,
        "Treasuries": -4,
        "Commodities": 4,
        "Cash": 0,
        "Gold": -2,
        "SPY Put Spread": -100
    },
    "Deflation": {
        "Stocks": -5,
        "Treasuries": 10,
        "Commodities": -10,
        "Cash": 0,
        "Gold": 5,
        "SPY Put Spread": 20
    }
}

# Streamlit UI
st.title("Macro Scenario-Based Portfolio Simulator")

st.header("Enter Scenario Probabilities")
probabilities = {}
total_prob = 0
for scenario in scenarios.keys():
    p = st.slider(f"{scenario} Probability (%)", min_value=0, max_value=100, value=25)
    probabilities[scenario] = p / 100
    total_prob += p

if total_prob != 100:
    st.warning("Total probabilities must sum to 100% to reflect reality.")

# Run simulation if total probability is valid
if total_prob == 100:
    df = pd.DataFrame(allocations)
    df["expected_return"] = 0.0

    for scenario, weight in probabilities.items():
        for asset in df["symbol"]:
            scenario_return = scenarios[scenario].get(asset, 0)
            df.loc[df["symbol"] == asset, "expected_return"] += weight * (scenario_return / 100)

    df["expected_dollar_return"] = df["allocation"] * df["expected_return"]
    df["final_value"] = df["allocation"] + df["expected_dollar_return"]

    portfolio_expected_return = df["expected_dollar_return"].sum() / df["allocation"].sum()
    portfolio_final_value = df["final_value"].sum()

    st.subheader("Simulation Results")
    st.dataframe(df.style.format({"allocation": "$ {:,.0f}", "expected_dollar_return": "$ {:,.0f}", "final_value": "$ {:,.0f}", "expected_return": "{:.2%}"}))
    st.metric("Expected Portfolio Return", f"{portfolio_expected_return:.2%}")
    st.metric("Expected Final Portfolio Value", f"$ {portfolio_final_value:,.0f}")

    st.subheader("Scenario Impact Summary")
    st.write(pd.DataFrame(scenarios))
