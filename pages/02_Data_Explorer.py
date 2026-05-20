import streamlit as st
import plotly.express as px
import pandas as pd
from data_loader import apply_global_style, load_data


def render():
    apply_global_style()
    
    st.title("📊 Data Explorer")
    
    st.markdown("""
Explore the climate and yield datasets to understand distributions, relationships, and patterns.
    """)
    
    data = load_data()
    
    # Tab-based exploration
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Climate Data", "Yield Data", "Relationships"])
    
    with tab1:
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crops", "4", "Maize, Rice, Cassava, Yam")
        with col2:
            st.metric("Regions", "6", "Geopolitical zones in Nigeria")
        with col3:
            st.metric("Years Covered", "25", "1999-2023")
        
        st.markdown("---")
        
        st.subheader("🌾 Food Security Risk by Region")
        
        food_security = data.get("food_security", pd.DataFrame()).copy()
        region_metrics = data.get("region_ensemble_metrics", pd.DataFrame()).copy()

        if not food_security.empty and "Food_Security_Risk_Score" in food_security.columns:
            fs_sorted = food_security.sort_values("Food_Security_Risk_Score", ascending=False)

            fig = px.bar(
                fs_sorted,
                x="Region",
                y="Food_Security_Risk_Score",
                title="Food Security Risk Score by Region",
                labels={"Food_Security_Risk_Score": "Risk Score"},
                color="Food_Security_Risk_Score",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fs_sorted, use_container_width=True)
        elif not region_metrics.empty and "Resilience_Index" in region_metrics.columns:
            region_sorted = region_metrics.sort_values("Resilience_Index", ascending=False)

            fig = px.bar(
                region_sorted,
                x="Region",
                y="Resilience_Index",
                title="Climate Resilience Index by Region",
                labels={"Resilience_Index": "Resilience Index"},
                color="Resilience_Index",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_sorted, use_container_width=True)
        else:
            st.info("No region-level food security or resilience data was found in the loaded results.")
    
    with tab2:
        st.subheader("🌡️ Climate Variables")
        
        st.markdown("""
Key climate variables used in the model:
- **Temperature**: Average seasonal temperature (°C)
- **Rainfall**: Annual and seasonal precipitation (mm)
- **Humidity**: Relative humidity (%)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Temperature Range**: 15-35°C across regions")
        with col2:
            st.info("**Rainfall Range**: 500-3000mm annually")
    
    with tab3:
        st.subheader("🌾 Crop Yield Data")

        if "crop_sensitivity" in data:
            crop_sensitivity = data['crop_sensitivity']
            if isinstance(crop_sensitivity, pd.DataFrame) and not crop_sensitivity.empty and {'Crop', 'Overall_Sensitivity'}.issubset(crop_sensitivity.columns):
                st.dataframe(crop_sensitivity, use_container_width=True)

                fig = px.bar(
                    crop_sensitivity,
                    x='Crop',
                    y='Overall_Sensitivity',
                    title="Crop Climate Sensitivity",
                    color='Overall_Sensitivity',
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No crop sensitivity table available or required columns ('Crop','Overall_Sensitivity') are missing.")
    
    with tab4:
        st.subheader("🔗 Climate-Yield Relationships")
        
        st.markdown("""
The TCN-MLP model learns complex relationships between climate sequences and crop yields:

- **Temporal Patterns**: How climate conditions during different growing periods affect final yield
- **Nonlinear Effects**: Interactions between multiple climate variables
- **Crop-Specific**: Each crop responds differently to the same climate variations
- **Regional Variations**: Soil and local practices create region-specific responses
        """)
        
        st.info("Use the **Model Architecture** page to understand how these relationships are captured.")


if __name__ == "__main__":
    render()
