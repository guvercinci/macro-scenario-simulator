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

# Reference values with editable trailing EPS and SPX
st.header("Trailing Market Assumptions")
col1, col2 = st.columns(2)
with col1:
    trailing_eps = st.number_input("Trailing SPX Earnings (EPS)", value=235.0, step=1.0)
with col2:
    current_spx = st.number_input("Current SPX Index Level", value=4700.0, step=10.0)

# Calculate P/E from SPX and EPS
trailing_pe = current_spx / trailing_eps
st.markdown(f"**Calculated Trailing P/E Ratio:** {trailing_pe:.2f}")

reference_prices = {
    "SPX": current_spx,
    "Gold": 2000,
    "Crude": 80,
    "10Y": 4.0,
    "Trailing_EPS": trailing_eps
}

# Default PE and EPS data
default_scenario_data = {
    "Recession": {"pe": 14, "eps_change": -0.20},
    "Stagflation": {"pe": 15, "eps_change": -0.10},
    "Boom": {"pe": 20, "eps_change": 0.15},
    "Deflation": {"pe": 17, "eps_change": -0.05}
}

# Editable assumption toggle
st.header("Scenario Assumption Configuration")
use_custom_assumptions = st.checkbox("Manually edit P/E and EPS change assumptions", value=False)

# Set up scenario assumptions either editable or default
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

# Assumed Gold prices per scenario
gold_targets = {
    "Recession": 2200,
    "Stagflation": 2500,
    "Boom": 1900,
    "Deflation": 2100
}

# Scenario probabilities
st.header("Enter Scenario Probabilities")
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
        data = scenario_data[scenario]
        pe = data["pe"]
        eps = reference_prices["Trailing_EPS"] * (1 + data["eps_change"])
        implied_spx_price = eps * pe
        gold_target = gold_targets[scenario]
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

    st.subheader("Scenario Assumptions Summary")
    summary_df = pd.DataFrame({
        "Scenario": scenario_names,
        "P/E Ratio": [scenario_data[s]["pe"] for s in scenario_names],
        "Earnings Change": [f"{scenario_data[s]['eps_change']*100:.0f}%" for s in scenario_names],
        "Implied SPX Price": [reference_prices["Trailing_EPS"] * (1 + scenario_data[s]["eps_change"]) * scenario_data[s]["pe"] for s in scenario_names],
        "Gold Price": [gold_targets[s] for s in scenario_names],
        "Crude Oil Price": [reference_prices["Crude"] * (0.9 if s == "Recession" else 1.2 if s == "Boom" else 1.0) for s in scenario_names],
        "10Y Yield": [reference_prices["10Y"] * (1.25 if s == "Boom" else 0.75 if s == "Recession" else 1.0) for s in scenario_names]
    })
    st.dataframe(summary_df.set_index("Scenario"))
