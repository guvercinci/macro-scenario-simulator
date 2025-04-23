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
