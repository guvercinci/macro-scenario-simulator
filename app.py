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

# Scenario inputs based on macro variables
st.header("Enter Scenario Macroeconomic Assumptions")
scenario_names = ["Recession", "Stagflation", "Boom", "Deflation"]
scenario_inputs = {}

for scenario in scenario_names:
    st.subheader(f"{scenario} Scenario")
    gdp_growth = st.number_input(f"GDP Growth (%) - {scenario}", value=-2.0 if scenario=="Recession" else 2.0, step=0.1, key=f"gdp_{scenario}")
    inflation = st.number_input(f"Inflation (%) - {scenario}", value=2.0, step=0.1, key=f"inflation_{scenario}")
    interest_rate = st.number_input(f"10Y Treasury Yield (%) - {scenario}", value=3.0, step=0.1, key=f"rate_{scenario}")
    spy_price = st.number_input(f"SPY Price - {scenario}", value=400, step=1, key=f"spy_{scenario}")
    crude_price = st.number_input(f"Crude Oil Price - {scenario}", value=80, step=1, key=f"crude_{scenario}")
    gold_price = st.number_input(f"Gold Price - {scenario}", value=2000, step=10, key=f"gold_{scenario}")

    scenario_inputs[scenario] = {
        "GDP": gdp_growth,
        "Inflation": inflation,
        "10Y": interest_rate,
        "SPY": spy_price,
        "Crude": crude_price,
        "Gold": gold_price
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

# Reference values for returns
reference_prices = {
    "SPY": 500,
    "Crude": 80,
    "10Y": 4.0,
    "Gold": 2000
}

# Run simulation
if total_prob == 100:
    df = editable_allocations.copy()
    df["expected_return"] = 0.0

    for scenario, weight in probabilities.items():
        for asset in df["symbol"]:
            inputs = scenario_inputs[scenario]
            if asset == "Stocks":
                r = (inputs["SPY"] / reference_prices["SPY"]) - 1
            elif asset == "Treasuries":
                r = (reference_prices["10Y"] - inputs["10Y"]) / reference_prices["10Y"]
            elif asset == "Commodities":
                crude_return = (inputs["Crude"] / reference_prices["Crude"]) - 1
                gold_return = (inputs["Gold"] / reference_prices["Gold"]) - 1
                r = 0.5 * (crude_return + gold_return)
            elif asset == "Gold":
                r = (inputs["Gold"] / reference_prices["Gold"]) - 1
            elif asset == "SPY Put Spread":
                r = (reference_prices["SPY"] - inputs["SPY"]) / reference_prices["SPY"] * 3
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

    st.subheader("Scenario Input Matrix")
    st.write(pd.DataFrame(scenario_inputs))
