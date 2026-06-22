import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from data_loader import apply_global_style, load_data


@st.cache_data(show_spinner=False)
def _load_fallback_crop_sensitivity_table() -> pd.DataFrame:
    seasonal_path = Path(__file__).resolve().parent.parent / "results" / "seasonal_climate_sensitivity_by_crop.csv"
    seasonal = pd.read_csv(seasonal_path) if seasonal_path.exists() else pd.DataFrame()
    if isinstance(seasonal, pd.DataFrame) and not seasonal.empty:
        seasonal = seasonal.copy()
        if "Crop" not in seasonal.columns:
            seasonal = seasonal.rename(columns={seasonal.columns[0]: "Crop"})
        numeric_cols = [col for col in seasonal.columns if col != "Crop"]
        if numeric_cols:
            for col in numeric_cols:
                seasonal[col] = pd.to_numeric(seasonal[col], errors="coerce")
            seasonal["Overall_Sensitivity"] = seasonal[numeric_cols].abs().mean(axis=1, skipna=True)
            return seasonal[["Crop", "Overall_Sensitivity"]]

    return pd.DataFrame()


def _get_crop_sensitivity_table(data: dict) -> pd.DataFrame:
    crop_sensitivity = data.get("crop_sensitivity", pd.DataFrame())
    if isinstance(crop_sensitivity, pd.DataFrame) and not crop_sensitivity.empty:
        if {"Crop", "Overall_Sensitivity"}.issubset(crop_sensitivity.columns):
            return crop_sensitivity

    return _load_fallback_crop_sensitivity_table()


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
            st.plotly_chart(fig, width='stretch')
            st.dataframe(fs_sorted, width='stretch')
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
            st.plotly_chart(fig, width='stretch')
            st.dataframe(region_sorted, width='stretch')
        else:
            st.info("No region-level food security or resilience data was found in the loaded results.")
    
    with tab2:
        st.subheader("🌡️ Climate Variables")
        
        st.markdown("""
    Key climate variables used in the model:
    - **Temperature**: Average seasonal temperature (°C)
    - **Rainfall**: Monthly total precipitation (PRECTOTCORR) in mm/month (annual totals are the sum of monthly values)
    - **Humidity**: Relative humidity (%)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Temperature Range**: 15-35°C across regions")
        with col2:
            st.info("**Typical annual rainfall**: 500–3000 mm (PRECTOTCORR stores monthly totals in mm/month)")
    
    with tab3:
        st.subheader("🌾 Crop Yield Data")

        processed = data.get("processed_dataset", pd.DataFrame())
        if isinstance(processed, pd.DataFrame) and not processed.empty:
            yield_col = None
            for candidate in ("Yield_kg_per_ha", "Yield", "yield"):
                if candidate in processed.columns:
                    yield_col = candidate
                    break

            if yield_col is not None and {"Crop", "Region"}.issubset(processed.columns):
                yield_summary = (
                    processed[["Crop", "Region", yield_col]]
                    .dropna(subset=[yield_col])
                    .groupby(["Crop", "Region"], as_index=False)
                    .agg(
                        Mean_Yield=(yield_col, "mean"),
                        Median_Yield=(yield_col, "median"),
                        Sample_Count=(yield_col, "count"),
                    )
                )

                st.caption("Observed yield data from the processed dataset.")
                st.dataframe(yield_summary, width='stretch', hide_index=True)

                fig = px.bar(
                    yield_summary,
                    x="Crop",
                    y="Mean_Yield",
                    color="Region",
                    barmode="group",
                    title="Average Yield by Crop and Region",
                    labels={"Mean_Yield": "Average Yield (kg/ha)"},
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Processed dataset is available, but no yield column was found.")

        else:
            st.info("No yield observations were found in the loaded results.")

#         crop_sensitivity = _get_crop_sensitivity_table(data)
#         if not crop_sensitivity.empty:
#             st.markdown("#### Crop Climate Sensitivity")
#             st.dataframe(crop_sensitivity, width='stretch', hide_index=True)

#             fig = px.bar(
#                 crop_sensitivity,
#                 x='Crop',
#                 y='Overall_Sensitivity',
#                 title="Crop Climate Sensitivity",
#                 color='Overall_Sensitivity',
#                 color_continuous_scale="Reds",
#             )
#             st.plotly_chart(fig, width='stretch')
#         else:
#             st.info("No crop sensitivity data could be resolved from the loaded results.")
    
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
