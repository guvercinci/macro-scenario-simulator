# app.py — Streamlit Macro Scenario-Based Portfolio Simulator (Single File)

import streamlit as st
import pandas as pd



# === Constants ===
MAX_MACRO_PE_IMPACT = 0.3
DEFAULT_CASH_YIELD = 0.04
BOND_DURATION = 7

# === Inputs ===
def valuation_inputs():
    st.sidebar.header("Valuation Inputs")
    eps = st.sidebar.number_input("Trailing SPX Earnings (EPS)", value=200.0, step=1.0)
    spx = st.sidebar.number_input("Current SPX Index Level", value=5300.0, step=10.0)
    return eps, spx

def macro_sliders():
    st.sidebar.header("Macro Conditions")
    liquidity = st.sidebar.slider("Liquidity", 0.0, 1.0, 0.5)
    fiscal = st.sidebar.slider("Fiscal Stimulus", 0.0, 1.0, 0.3)
    geo = st.sidebar.slider("Geopolitical Risk", 0.0, 1.0, 0.1)
    return liquidity, fiscal, geo

def get_macro_multiplier(liquidity, fiscal, geo):
    return 1 + min(MAX_MACRO_PE_IMPACT, liquidity * 0.10 + fiscal * 0.05 - geo * 0.07)

def get_scenario_data():
    return {
        "Recession": {"pe": 14, "eps_change": -0.20},
        "Stagflation": {"pe": 15, "eps_change": -0.10},
        "Boom": {"pe": 20, "eps_change": 0.15},
        "Deflation": {"pe": 17, "eps_change": -0.05},
    }

def scenario_probabilities(auto, liquidity, fiscal, geo):
    st.sidebar.header("Scenario Probabilities")
    if auto:
        if liquidity > 0.6 and fiscal > 0.5 and geo < 0.3:
            probs = {"Boom": 50, "Stagflation": 15, "Recession": 15, "Deflation": 20}
        elif liquidity < 0.3 and fiscal < 0.3:
            probs = {"Boom": 5, "Stagflation": 30, "Recession": 40, "Deflation": 25}
        elif geo > 0.6:
            probs = {"Boom": 10, "Stagflation": 35, "Recession": 35, "Deflation": 20}
        else:
            probs = {"Boom": 25, "Stagflation": 25, "Recession": 25, "Deflation": 25}
    else:
        probs = {}
        total = 0
        for s in ["Recession", "Stagflation", "Boom", "Deflation"]:
            p = st.sidebar.number_input(f"{s} %", min_value=0, max_value=100, value=25)
            probs[s] = p
            total += p
        if total != 100:
            st.warning("Scenario probabilities must sum to 100%.")
            return probs, False
    return probs, True

def calculate_asset_targets(liquidity, fiscal, geo):
    gold = 2000 * (1 + liquidity * 0.05 + geo * 0.1)
    crude = 80 * (1 + fiscal * 0.1)
    bond = 4.0 * (1 - liquidity * 0.05 - geo * 0.05 + fiscal * 0.05)
    return {"Gold": gold, "Crude": crude, "10Y": bond}

def portfolio_editor():
    st.subheader("Portfolio Allocation")
    df = pd.DataFrame({
        "symbol": ["Stocks", "Treasuries", "Commodities", "Cash", "Gold", "SPY Put Spread"],
        "allocation": [250000, 30000, 20000, 50000, 10000, 10000]
    })
    return st.data_editor(df, num_rows="dynamic", use_container_width=True)

def run_simulation(eps, spx, probs, scenarios, macro_mult, df, targets):
    total_alloc = df["allocation"].sum()
    weighted_eps = sum(probs[s]/100 * eps * (1 + scenarios[s]['eps_change']) for s in probs)
    weighted_pe = sum(probs[s]/100 * scenarios[s]['pe'] for s in probs)
    adj_pe = weighted_pe * macro_mult
    fair_spx = weighted_eps * adj_pe
    df["expected_return"] = 0.0
    for s, weight in probs.items():
        implied_spx = eps * (1 + scenarios[s]['eps_change']) * scenarios[s]['pe'] * macro_mult
        bond_yield = targets["10Y"]
        for i, row in df.iterrows():
            asset = row["symbol"]
            if asset == "Stocks":
                r = (implied_spx / spx) - 1
            elif asset == "Treasuries":
                r = BOND_DURATION * (targets["10Y"] - bond_yield) / 100
            elif asset == "Commodities":
                crude_r = (targets["Crude"] / 80) - 1
                gold_r = (targets["Gold"] / 2000) - 1
                r = 0.5 * (crude_r + gold_r)
            elif asset == "Gold":
                r = (targets["Gold"] / 2000) - 1
            elif asset == "Cash":
                r = DEFAULT_CASH_YIELD
            elif asset == "SPY Put Spread":
                fall = (spx - implied_spx) / spx
                r = min(max(fall, 0), 0.2) * 3
            else:
                r = 0
            df.at[i, "expected_return"] += (weight / 100) * r
    df["expected_dollar_return"] = df["allocation"] * df["expected_return"]
    df["final_value"] = df["allocation"] + df["expected_dollar_return"]
    return df, {
        "expected_return": f"{df['expected_dollar_return'].sum() / total_alloc:.2%}",
        "final_value": round(df["final_value"].sum())
    }, fair_spx

def main():
    auto = st.sidebar.checkbox("Auto-adjust scenario probabilities", value=True)
    st.set_page_config(page_title="Macro Portfolio Simulator", layout="wide")
    st.title("Macro Scenario-Based Portfolio Simulator")
    st.caption("Simulate how portfolios perform across macro regimes—like a hedge fund would.")

    eps, spx = valuation_inputs()
    liquidity, fiscal, geo = macro_sliders()
    macro_mult = get_macro_multiplier(liquidity, fiscal, geo)
    probs, valid = scenario_probabilities(auto=auto, liquidity=liquidity, fiscal=fiscal, geo=geo)
    if not valid:
        st.stop()
    scenarios = get_scenario_data()
    targets = calculate_asset_targets(liquidity, fiscal, geo)
    alloc_df = portfolio_editor()
    result_df, metrics, fair_spx = run_simulation(eps, spx, probs, scenarios, macro_mult, alloc_df, targets)

    st.subheader("Simulation Results")
    st.dataframe(result_df.style.format({"allocation": "$ {:,.0f}", "expected_dollar_return": "$ {:,.0f}", "final_value": "$ {:,.0f}", "expected_return": "{:.2%}"}))
    st.metric("Expected Portfolio Return", metrics["expected_return"])
    st.metric("Expected Final Portfolio Value", f"$ {metrics['final_value']:,}")
    st.markdown(f"**Fair SPX Estimate:** {fair_spx:,.0f}")

if __name__ == "__main__":
    main()
