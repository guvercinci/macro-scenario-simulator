# Macro Scenario-Based Portfolio Simulator with Web UI (Streamlit)

import streamlit as st
import pandas as pd

st.title("Macro Scenario-Based Portfolio Simulator")
st.markdown("Build a resilient, macro-aware portfolio that performs across economic regimes—just like a risk-parity hedge fund would.")
st.markdown("""
This tool simulates portfolio returns based on macroeconomic scenarios.
It uses assumptions about SPX valuation, earnings sensitivity, commodity prices,
interest rate shifts, and macro factors inspired by Ray Dalio's framework.

**How macro factors influence assets:**
- **Liquidity**: High liquidity boosts equity valuations and gold prices, and depresses yields.
- **Fiscal Stimulus**: Increases earnings expectations, equity multiples, and demand for energy.
- **Geopolitical Risk**: Suppresses equity P/E ratios, boosts gold (safe haven), and lowers bond yields (flight to safety).
""")

# 1. Market Valuation Inputs
st.header("Current Market Valuation")
col1, col2 = st.columns(2)
with col1:
    trailing_eps = st.number_input("Trailing SPX Earnings (EPS)", value=200.27, step=1.0)
with col2:
    current_spx = st.number_input("Current SPX Index Level", value=5287.76, step=10.0)

trailing_pe = current_spx / trailing_eps
st.markdown(f"**Calculated Trailing P/E Ratio:** {trailing_pe:.2f}")

# 2. Macro backdrop and scenario probabilities
probabilities = {}  # Move this up to ensure it's accessible regardless of toggle
st.markdown("""
### Adjust Simulation Parameters
Toggle automatic scenario assignment based on macro conditions.
""")
auto_prob_toggle = st.checkbox("Auto-adjust scenario probabilities based on macro inputs", value=False)
col_a, col_b = st.columns([1, 2])


with col_a:
    st.subheader("Scenario Probabilities")
    default_scenario_data = {
        "Recession": {"pe": 14, "eps_change": -0.20},
        "Stagflation": {"pe": 15, "eps_change": -0.10},
        "Boom": {"pe": 20, "eps_change": 0.15},
        "Deflation": {"pe": 17, "eps_change": -0.05}
    }
    scenario_names = list(default_scenario_data.keys())
    probabilities = {}
    total_prob = 0

    if not auto_prob_toggle:
        for scenario in scenario_names:
            p = st.number_input(f"{scenario} (manual)", min_value=0, max_value=100, value=25, step=1, key=f"manual_prob_{scenario}")
            probabilities[scenario] = p
            total_prob += p
        st.markdown(f"**Total Probability:** {total_prob:.1f}%")
        if total_prob != 100:
            st.warning("Total probabilities must sum to 100%.")
    

with col_b:
    st.subheader("Global Macro Backdrop")
    st.caption("These sliders simulate monetary, fiscal, and geopolitical dynamics. Their values adjust asset class sensitivity based on Ray Dalio's macro playbook.")
    col3, col4, col5 = st.columns(3)
    with col3:
        liquidity_index = st.slider("Liquidity (0=Drained, 1=Flooded)", 0.0, 1.0, 0.5)
    with col4:
        fiscal_stimulus = st.slider("Fiscal Stimulus (0=None, 1=Extreme)", 0.0, 1.0, 0.3)
    with col5:
        geopolitical_risk = st.slider("Geopolitical Risk (0=Stable, 1=Crisis)", 0.0, 1.0, 0.1)

# 3. Macro-Adjusted Anchors
st.subheader("Macro-Adjusted Asset Anchors")
adjusted_gold = 2000 * (1 + liquidity_index * 0.05 + geopolitical_risk * 0.1)
adjusted_crude = 80 * (1 + fiscal_stimulus * 0.1)
adjusted_10y = 4.0 * (1 - liquidity_index * 0.05 - geopolitical_risk * 0.05 + fiscal_stimulus * 0.05)

multiplier_explainer = liquidity_index * 0.1 + fiscal_stimulus * 0.05 - geopolitical_risk * 0.05
spx_macro_adjustment = trailing_eps * trailing_pe * (1 + multiplier_explainer)

st.markdown(f"**Implied SPX (Current Valuation):** {trailing_eps * trailing_pe:,.0f}")


st.markdown(f"**Gold Price:** ${adjusted_gold:.2f}  Crude Oil:** ${adjusted_crude:.2f}  10-Year Yield:** {adjusted_10y:.2f}%")

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

# --- AUTO-IMPROVEMENT LOGIC ---

# Infer scenario probabilities from macro backdrop (optional override of manual sliders)
def infer_probabilities_from_macro(liq, fiscal, geo):
    if liq > 0.6 and fiscal > 0.5 and geo < 0.3:
        return {"Boom": 50, "Stagflation": 15, "Recession": 15, "Deflation": 20}
    elif liq < 0.3 and fiscal < 0.3:
        return {"Boom": 5, "Stagflation": 30, "Recession": 40, "Deflation": 25}
    elif geo > 0.6:
        return {"Boom": 10, "Stagflation": 35, "Recession": 35, "Deflation": 20}
    else:
        return {"Boom": 25, "Stagflation": 25, "Recession": 25, "Deflation": 25}

# Apply auto probabilities if toggled
if auto_prob_toggle:
    raw_probs = infer_probabilities_from_macro(liquidity_index, fiscal_stimulus, geopolitical_risk)
    scaled = {k: round(v * 100 / sum(raw_probs.values())) for k, v in raw_probs.items()}
    total_prob = sum(scaled.values())

    if total_prob != 100:
        diff = 100 - total_prob
        max_key = max(scaled, key=scaled.get)
        scaled[max_key] += diff

    probabilities = scaled
    total_prob = sum(probabilities.values())
    st.markdown(f"**Total Probability:** {total_prob:.1f}%")
    for k in scenario_names:
        st.markdown(f"**{k}:** {probabilities.get(k, 0)}%")
    st.markdown(f"**Total Probability:** {total_prob:.1f}%")
    for k, v in probabilities.items():
        st.markdown(f"**{k}:** {v}%")

# --- END AUTO-IMPROVEMENT SECTION ---

# 4. Scenario assumptions
st.header("Scenario Assumptions")
use_custom_assumptions = st.checkbox("Edit Scenario Multiples and EPS Impact", value=False)
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

# Calculate scenario-weighted fair value
weighted_eps = sum(
    probabilities[s] / 100 * trailing_eps * (1 + scenario_data[s]['eps_change'])
    for s in scenario_names
)
weighted_pe = sum(
    probabilities[s] / 100 * scenario_data[s]['pe']
    for s in scenario_names
)
macro_pe_liquidity = liquidity_index * 0.10
macro_pe_fiscal = fiscal_stimulus * 0.05
macro_pe_geo = geopolitical_risk * -0.07
macro_pe_multiplier = 1 + macro_pe_liquidity + macro_pe_fiscal + macro_pe_geo
adjusted_weighted_pe = weighted_pe * macro_pe_multiplier
spx_fair_value = weighted_eps * adjusted_weighted_pe
st.markdown("**Fair SPX = EPS × P/E:**")
st.markdown(f"→ Scenario-Weighted EPS = {weighted_eps:.2f}")
st.markdown(f"→ Scenario-Weighted P/E = {weighted_pe:.2f}")
st.markdown(f"→ Macro-Adjusted P/E = {weighted_pe:.2f} × {macro_pe_multiplier:.3f} = {adjusted_weighted_pe:.2f}")
st.markdown(f"→ **Macro-Adjusted Fair Value = {weighted_eps:.2f} × {adjusted_weighted_pe:.2f} = {spx_fair_value:,.0f}**")


st.markdown(f"**Fair SPX = EPS × P/E:**")
st.markdown(f"→ Scenario-Weighted EPS = {weighted_eps:.2f}")

# Show component breakdown
with st.expander("See Fair Value Calculation Breakdown"):
    st.markdown("### Weighted EPS and P/E by Scenario")
    calc_df = pd.DataFrame([
        {
            "Scenario": s,
            "Probability (%)": probabilities[s],
            "Earnings Estimate": trailing_eps * (1 + scenario_data[s]['eps_change']),
            "P/E Ratio": scenario_data[s]['pe'],
            "Weighted EPS": probabilities[s] / 100 * trailing_eps * (1 + scenario_data[s]['eps_change']),
            "Weighted P/E": probabilities[s] / 100 * scenario_data[s]['pe']
        } for s in scenario_names
    ])
    st.dataframe(calc_df.set_index("Scenario").style.format({
        "Probability (%)": "{:.0f}%",
        "Earnings Estimate": "{:.2f}",
        "P/E Ratio": "{:.2f}",
        "Weighted EPS": "{:.2f}",
        "Weighted P/E": "{:.2f}"
    }))
    st.markdown(f"**Total Weighted EPS:** {weighted_eps:.2f}")
    st.markdown(f"**Total Weighted P/E:** {weighted_pe:.2f}")
    st.markdown(f"**Fair Value = EPS × P/E = {weighted_eps:.2f} × {weighted_pe:.2f} = {spx_fair_value:,.0f}**")
st.dataframe(pd.DataFrame(default_scenario_data).T.rename(columns={"pe": "P/E Ratio", "eps_change": "Earnings Change"}).style.format({"Earnings Change": "{:.0%}"}))

# 5. Portfolio allocations
st.header("Adjust Portfolio Allocations")
alloc_df = pd.DataFrame(
    {
        "symbol": ["Stocks", "Treasuries", "Commodities", "Cash", "Gold", "SPY Put Spread"],
        "allocation": [234434, 30000, 20000, 47000, 10000, 9260]
    }
)
editable_allocations = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

# 6. Simulation results
if total_prob == 100:
    df = editable_allocations.copy()
    df["expected_return"] = 0.0

    for scenario, weight in probabilities.items():
        pe = scenario_data[scenario]["pe"]
        eps = reference_prices["Trailing_EPS"] * (1 + scenario_data[scenario]["eps_change"])
        implied_spx_price = eps * pe

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

    st.header("Simulation Results")
    st.caption("Results shown below reflect the impact of scenario probabilities, macro environment, and your portfolio allocation. Grouped into strategic exposures:")

    risk_buckets = {
        "Growth Risk": ["Stocks"],
        "Inflation Hedge": ["Commodities", "Gold"],
        "Deflation Hedge": ["Treasuries", "SPY Put Spread"],
        "Neutral": ["Cash"]
    }

    def classify_risk(symbol):
        for bucket, symbols in risk_buckets.items():
            if symbol in symbols:
                return bucket
        return "Other"

    df["Exposure Type"] = df["symbol"].apply(classify_risk)
    st.dataframe(df.style.format({"allocation": "$ {:,.0f}", "expected_dollar_return": "$ {:,.0f}", "final_value": "$ {:,.0f}", "expected_return": "{:.2%}"}))
    st.metric("Expected Portfolio Return", f"{df['expected_dollar_return'].sum() / df['allocation'].sum():.2%}")
    st.metric("Expected Final Portfolio Value", f"$ {df['final_value'].sum():,.0f}")
