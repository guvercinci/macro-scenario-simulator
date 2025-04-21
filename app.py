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

# Editable macro scenario returns
st.header("Edit Scenario Return Assumptions (%)")
assets = list(editable_allocations["symbol"])
scenario_names = ["Recession", "Stagflation", "Boom", "Deflation"]

# Default values for the return matrix
default_returns = {
    "Recession": [-15, 5, -5, 0, 3, 207],
    "Stagflation": [-10, -8, 6, 0, 10, 100],
    "Boom": [12, -4, 4, 0, -2, -100],
    "Deflation": [-5, 10, -10, 0, 5, 20]
}

scenario_df = pd.DataFrame(default_returns, index=assets)
editable_returns = st.data_editor(scenario_df, use_container_width=True)

# Scenario probabilities
st.header("Enter Scenario Probabilities")
probabilities = {}
total_prob = 0
for scenario in editable_returns.columns:
    p = st.slider(f"{scenario} Probability (%)", min_value=0, max_value=100, value=25)
    probabilities[scenario] = p
    total_prob += p

st.write(f"**Total Assigned Probability:** {total_prob:.1f}%")

if total_prob != 100:
    st.warning("Total probabilities must sum to 100% to reflect reality.")

# Run simulation if total probability is valid
if total_prob == 100:
    df = editable_allocations.copy()
    df["expected_return"] = 0.0

    for scenario, weight in probabilities.items():
        for asset in df["symbol"]:
            scenario_return = editable_returns.loc[asset, scenario]
            df.loc[df["symbol"] == asset, "expected_return"] += (weight / 100) * (scenario_return / 100)

    df["expected_dollar_return"] = df["allocation"] * df["expected_return"]
    df["final_value"] = df["allocation"] + df["expected_dollar_return"]

    portfolio_expected_return = df["expected_dollar_return"].sum() / df["allocation"].sum()
    portfolio_final_value = df["final_value"].sum()

    st.subheader("Simulation Results")
    st.dataframe(df.style.format({"allocation": "$ {:,.0f}", "expected_dollar_return": "$ {:,.0f}", "final_value": "$ {:,.0f}", "expected_return": "{:.2%}"}))
    st.metric("Expected Portfolio Return", f"{portfolio_expected_return:.2%}")
    st.metric("Expected Final Portfolio Value", f"$ {portfolio_final_value:,.0f}")

    st.subheader("Scenario Impact Matrix")
    st.dataframe(editable_returns.style.format("{:.1f}%"))
