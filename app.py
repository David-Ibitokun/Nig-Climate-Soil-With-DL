import streamlit as st
import plotly.express as px

from data_loader import apply_global_style, load_data


def main():
    apply_global_style()

    data = load_data()
    meta = data.get("metadata") or {}
    perf = meta.get("performance", {}) if isinstance(meta, dict) else {}

    st.title("🌾 Climate Change & Food Security in Nigeria")
    st.markdown(
        """
**Evaluating the impact of climate change and soil on food security** using a TCN–MLP hybrid model.

Use the **left sidebar** to open detailed pages (Model Performance, Crop Analysis, Regional Vulnerability, etc.).
"""
    )

    # Top-line metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test R²", f"{perf.get('final_test_r2', 0):.4f}")
    with col2:
        st.metric("Test MAE", f"{perf.get('final_test_mae', 0):.1f} kg/ha")
    with col3:
        st.metric("Crops", "4", "Maize, Rice, Cassava, Yam")
    with col4:
        st.metric(
            "Temporal features",
            f"{(meta.get('data', {}) or {}).get('n_temporal_features', 0)}",
        )

    # Key findings
    st.subheader("📈 Key Findings")
    left, right = st.columns(2)
    with left:
        st.info(
            """
**🌾 Crop performance**
- Analysis covers **Maize, Rice, Cassava, and Yam**
- Yield variability across regions suggests strong climate sensitivity
- See **Crop Analysis** for sensitivity breakdowns
"""
        )
    with right:
        st.warning(
            """
**🌡️ Climate vulnerability**
- Some regions show high food-security risk under extreme scenarios
- Warming and drought can reduce yields significantly for sensitive crops
- See **Regional Vulnerability** for heatmaps and scenario impacts
"""
        )

    st.markdown("---")

    # Food security overview
    st.subheader("📊 General Food Security Assessment")
    food_security = data["food_security"].copy()
    fs_sorted = food_security.sort_values("Food_Security_Risk_Score", ascending=False)

    fig = px.bar(
        fs_sorted,
        x="Region",
        y="Food_Security_Risk_Score",
        title="Food Security Risk Score by Region",
        labels={"Food_Security_Risk_Score": "Risk Score (Higher is Worse)"},
        color="Food_Security_Risk_Score",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fs_sorted, use_container_width=True)

    st.caption(
        "All values shown in this app are loaded from artifacts produced by "
        "`New_Changes/yield_changes/Combined.ipynb` (saved under `New_Changes/results/` and `New_Changes/models/`)."
    )

    st.markdown("---")
    
    # Explanation of Food Security Risk Score
    st.subheader("📋 How is the Food Security Risk Score Calculated?")
    
    with st.expander("ℹ️ Click to learn about the risk scoring methodology"):
        st.markdown("""
### Risk Score Formula

The **Food Security Risk Score** combines three climate stress indicators into a single 0–1 scale (higher = worse):

$$\\text{Risk Score} = 0.4 \\times R_{\\text{extreme}} + 0.3 \\times R_{\\text{drought}} + 0.3 \\times R_{\\text{stability}}$$

---

### **Component 1: Extreme Scenario Risk (40% weight)**
Reflects yield loss under compound 2°C warming + 30% drought:
- **Score 1.0**: Yield drops ≥15% → Critical threat
- **Score 0.5**: Yield drops 5–15% → Moderate threat  
- **Score 0.0**: Yield drops <5% → Low threat

*Reasoning*: Compound stress is most threatening; gets highest weight.

---

### **Component 2: Drought Scenario Risk (30% weight)**
Reflects yield loss under -40% rainfall alone:
- **Score 1.0**: Yield drops ≥20% → Severe drought impact
- **Score 0.5**: Yield drops 10–20% → Moderate impact
- **Score 0.0**: Yield drops <10% → Minor impact

*Reasoning*: Drought is the #1 climate threat in Nigeria; major but slightly less severe than compound stress.

---

### **Component 3: Stability Deterioration (30% weight)**
Measures whether yields become more volatile under stress:
- **Ratio**: (yield volatility under extreme) / (baseline volatility)
- **Score**: 0.3 × ratio (capped at 1.0)

*Reasoning*: Volatile yields prevent farmers from planning; unpredictability undermines long-term food security.

---

### **Risk Interpretation**

| Score | Category | Policy Response |
|-------|----------|-----------------|
| 0.70–1.0 | **CRITICAL** | Immediate irrigation, drought-resistant seeds, early warning systems |
| 0.50–0.69 | **HIGH** | Build adaptive capacity; invest in resilient varieties |
| 0.30–0.49 | **MODERATE** | Monitor; support gradual transition to climate-smart practices |
| 0.0–0.29 | **LOW** | Focus on yield improvement; minimal adaptation urgency |

---

### **Why This Formula?**

- **Not universal**: Weights (0.4/0.3/0.3) are custom-designed for Nigeria based on agronomic literature and policy priorities.
- **Transparent**: Each component is interpretable; thresholds align with farming practices.
- **Actionable**: Scores directly map to policy interventions.
- **Validated**: Rankings match historical drought impacts (2011–2015 hit South-West hardest).

See **Regional Vulnerability** page for detailed breakdowns by region.
        """)

    st.caption(
        "For more information, see: `FOOD_SECURITY_RISK_METHODOLOGY.md` in project root."
    )


if __name__ == "__main__":
    main()
