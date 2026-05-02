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


if __name__ == "__main__":
    main()
