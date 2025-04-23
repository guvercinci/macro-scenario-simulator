        "Crude": 80 * (1 + fiscal * 0.2 + geo * 0.3 + liq * 0.1),
        "10Y": 4.0 * (1 - liq * 0.05 - geo * 0.05 + fiscal * 0.05),
    }

def portfolio_editor():
    equity_beta = st.sidebar.slider("Equity Beta (vs SPX)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Set how sensitive Equities are to SPX changes. 1.0 = market beta")
    st.subheader("Portfolio Allocation")
    portfolio_size = st.number_input("Total Portfolio Size ($)", value=100000, step=10000)
    preset = st.selectbox("Choose a Portfolio Preset:", ["Balanced (60/40)", "Aggressive Growth", "Defensive Hedge", "Custom"], index=0)

    if preset == "Balanced (60/40)":
        data = {"symbol": ["Equities", "Fixed Income", "Cash", "Commodities", "Gold", "Hedging Instruments"], "allocation_pct": [60, 30, 5, 2.5, 2.5, 0]}
    elif preset == "Aggressive Growth":
        data = {"symbol": ["Equities", "Fixed Income", "Cash", "Commodities", "Gold", "Hedging Instruments"], "allocation_pct": [80, 10, 2, 5, 2, 1]}
    elif preset == "Defensive Hedge":
        data = {"symbol": ["Equities", "Fixed Income", "Cash", "Commodities", "Gold", "Hedging Instruments"], "allocation_pct": [30, 30, 10, 10, 10, 10]}
    else:
        data = {"symbol": ["Equities", "Fixed Income", "Cash", "Commodities", "Gold", "Hedging Instruments"], "allocation_pct": [50, 30, 10, 5, 3, 2]}

    df = pd.DataFrame(data)
    df = st.data_editor(df, num_rows="fixed", use_container_width=True, key="portfolio_editor")
    total_pct = df["allocation_pct"].sum()
    st.markdown(f"**Total Allocation: {total_pct:.1f}%**")
    if abs(total_pct - 100) > 0.1:
        st.error("Total portfolio allocation must equal 100%. Please adjust your percentages.")
        st.stop()
    df["allocation"] = df["allocation_pct"] / 100 * portfolio_size
    df["allocation"] = df["allocation_pct"] / 100 * 100000
    df["beta"] = df["symbol"].apply(lambda x: equity_beta if x == "Equities" else 1.0)
    return df[["symbol", "allocation", "beta"]]

def simulate(eps, spx, probs, scenarios, macro_mult, alloc, targets):
    total = alloc["allocation"].sum()
    weighted_eps = sum(probs[s]/100 * eps * (1 + scenarios[s]['eps_change']) for s in probs)
    weighted_pe = sum(probs[s]/100 * scenarios[s]['pe'] for s in probs)
    fair_spx = weighted_eps * weighted_pe * macro_mult

    alloc["expected_return"] = 0.0
    for s, weight in probs.items():
        implied_spx = eps * (1 + scenarios[s]['eps_change']) * scenarios[s]['pe'] * macro_mult
        for i, row in alloc.iterrows():
            asset = row["symbol"]
            if asset == "Equities":
                r = ((implied_spx / spx) - 1) * row.get("beta", 1.0)
            elif asset == "Fixed Income":
                r = 0  # No change in bond yield, assuming flat environment; update if dynamic yield logic added
            elif asset == "Commodities":
                r = 0.5 * ((targets["Crude"] / 80 - 1) + (targets["Gold"] / 2000 - 1))
            elif asset == "Gold":
                r = (targets["Gold"] / 2000) - 1
            elif asset == "Cash":
                r = DEFAULT_CASH_YIELD
            elif asset == "Hedging Instruments":
                fall = (spx - implied_spx) / spx if spx != 0 else 0
                r = min(max(fall * 3, -0.1), 0.3)  # Hedge return scaled by 3x inverse SPX change, capped between -10% and +30%
            else:
                r = 0
            alloc.at[i, "expected_return"] += (weight / 100) * r

    alloc["expected_dollar_return"] = alloc["allocation"] * alloc["expected_return"]
    alloc["final_value"] = alloc["allocation"] + alloc["expected_dollar_return"]
    return alloc, fair_spx, weighted_eps, weighted_pe

def main():
    st.title("Macro Scenario-Based Portfolio Simulator")
    st.markdown("""
This tool helps simulate how a diversified portfolio might respond across different macroeconomic scenarios.
We combine scenario-weighted earnings (EPS) and valuation (P/E ratio), then apply macro adjustments to estimate SPX fair value.
Asset class returns are then computed based on these conditions.
""")

    eps, spx = valuation_inputs()
    liq, fiscal, geo = macro_conditions()
    auto = st.sidebar.checkbox("Auto-adjust scenario probabilities", value=True)
    probs, valid = scenario_probabilities(auto, liq, fiscal, geo)
    if not valid:
        st.warning("Scenario probabilities must sum to 100%.")
        st.stop()

    scenarios = default_scenarios()
    macro_mult = macro_multiplier(liq, fiscal, geo)
    targets = macro_targets(liq, fiscal, geo)
    alloc = portfolio_editor()
    results, fair_spx, weighted_eps, weighted_pe = simulate(eps, spx, probs, scenarios, macro_mult, alloc, targets)

    st.subheader("Calculation Summary")
    trailing_pe = spx / eps if eps != 0 else 0
    st.markdown(f"**1. Trailing P/E (SPX / EPS):** {spx:,.0f} / {eps:.2f} = {trailing_pe:.2f}")
    st.markdown("_This shows how expensive the market is relative to trailing earnings._")
    st.markdown(f"**2. Weighted Forward EPS:** {weighted_eps:.2f}")
    st.markdown("_This forecasts expected earnings based on how likely each scenario is to occur._")
    st.markdown(f"**3. Weighted P/E:** {weighted_pe:.2f}")
    st.markdown("_This is the average scenario-based valuation multiple, weighted by macro scenario probabilities._")
    st.markdown(f"**4. Macro Multiplier:** {macro_mult:.3f}")
    st.markdown("_Reflects the combined effect of liquidity, stimulus, and geopolitical risk on valuations._")
    st.markdown(f"**5. Fair SPX Estimate:** {weighted_eps:.2f} × {weighted_pe:.2f} × {macro_mult:.3f} = {fair_spx:,.0f}")
    st.markdown("_The final macro-adjusted fair value for the S&P 500 based on fundamentals and macro overlays._")

    st.subheader("Macro-Adjusted Asset Targets")
    st.markdown(f"- **Gold Target Price:** ${targets['Gold']:.2f}")
    st.markdown(f"- **Crude Oil Target Price:** ${targets['Crude']:.2f}")
    st.markdown(f"- **10-Year Yield Estimate:** {targets['10Y']:.2f}%")

    st.subheader("Simulation Results")
    st.dataframe(results.style.format({
        "allocation": "$ {:,.0f}",
        "expected_dollar_return": "$ {:,.0f}",
        "final_value": "$ {:,.0f}",
        "expected_return": "{:.2%}"
    }))
    st.metric("Fair SPX Estimate", f"{fair_spx:,.0f}")
    st.metric("Expected Portfolio Return", f"{results['expected_dollar_return'].sum() / results['allocation'].sum():.2%}")
    st.metric("Expected Final Value", f"$ {results['final_value'].sum():,.0f}")

if __name__ == "__main__":
    main()
