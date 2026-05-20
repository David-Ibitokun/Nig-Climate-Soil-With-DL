import streamlit as st
import pandas as pd
import numpy as np
from data_loader import apply_global_style, load_data


def render():
    apply_global_style()
    
    st.title("🎯 Make Prediction")
    
    st.markdown("""
Use the TCN-MLP Ensemble model to predict crop yields based on climate parameters.
Select your inputs and the model will provide yield predictions with confidence intervals.
    """)
    
    data = load_data()
    processed_dataset = data.get("processed_dataset") if isinstance(data.get("processed_dataset"), pd.DataFrame) else pd.DataFrame()

    if not processed_dataset.empty:
        crop_options = sorted(processed_dataset["Crop"].dropna().unique().tolist())
        region_options = sorted(processed_dataset["Region"].dropna().unique().tolist())
        year_min, year_max = 2026, 2030
    else:
        crop_options = ["Maize", "Rice", "Cassava", "Yam"]
        region_options = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
        year_min, year_max = 2026, 2030

    def _series_stats(column_name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        if processed_dataset.empty or column_name not in processed_dataset.columns:
            return default

        values = processed_dataset[column_name].dropna()
        if values.empty:
            return default

        return (
            float(values.quantile(0.05)),
            float(values.median()),
            float(values.quantile(0.95)),
        )

    temp_min, temp_default, temp_max = _series_stats("T2M_m1", (23.3, 25.7, 30.5))
    rainfall_min, rainfall_default, rainfall_max = _series_stats("PRECTOTCORR_m1", (0.0, 540.7, 2068.0))
    humidity_min, humidity_default, humidity_max = _series_stats("RH2M_m1", (21.6, 79.9, 90.7))
    
    # Prediction interface
    st.subheader("📋 Select Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop = st.selectbox(
            "Select Crop",
            crop_options
        )
    
    with col2:
        region = st.selectbox(
            "Select Region",
            region_options
        )
    
    with col3:
        year = st.slider(
            "Select Year",
            min_value=year_min,
            max_value=year_max,
            value=year_max
        )
    
    st.markdown("---")
    
    st.subheader("🌡️ Climate Reference Inputs")

    st.caption(
        "The trained model uses a 12-month climate sequence. These sliders provide a coarse historical reference profile based on the processed dataset."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.slider(
            "Reference Monthly Mean Temperature (°C)",
            min_value=float(round(temp_min, 1)),
            max_value=float(round(temp_max, 1)),
            value=float(round(temp_default, 1)),
            step=0.5
        )
    
    with col2:
        rainfall = st.slider(
            "Reference Monthly Rainfall (mm)",
            min_value=float(round(rainfall_min, 1)),
            max_value=float(round(rainfall_max, 1)),
            value=float(round(rainfall_default, 1)),
            step=10.0
        )
    
    with col3:
        humidity = st.slider(
            "Reference Relative Humidity (%)",
            min_value=float(round(humidity_min, 1)),
            max_value=float(round(humidity_max, 1)),
            value=float(round(humidity_default, 1)),
            step=1.0
        )

    with st.expander("Why these ranges?", expanded=False):
        st.markdown(
            f"""
These controls are anchored to the processed dataset rather than arbitrary guesses.

- **Year**: {year_min} to {year_max}, set as a future projection window.
- **Temperature**: observed monthly mean-temperature distribution from the dataset.
- **Rainfall**: observed monthly precipitation distribution from the dataset.
- **Humidity**: observed monthly relative humidity distribution from the dataset.

The current page is still a coarse scenario interface. The real network consumes 12 monthly values for 9 climate features, so a full inference form would need monthly sequence inputs rather than a single summary per variable.
            """
        )
    
    st.markdown("---")
    
    # Make prediction
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        st.info("Illustrative prediction output shown here. The page still needs the full model inference pipeline to return a real ensemble forecast.")
        
        # Placeholder for actual prediction
        predicted_yield = np.random.normal(2500, 300)
        confidence_lower = predicted_yield - 200
        confidence_upper = predicted_yield + 200
        
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Yield", f"{predicted_yield:.0f} kg/ha")
        with col2:
            st.metric("Lower Bound (95%)", f"{confidence_lower:.0f} kg/ha")
        with col3:
            st.metric("Upper Bound (95%)", f"{confidence_upper:.0f} kg/ha")
        
        st.markdown("---")
        
        st.subheader("📈 Prediction Details")
        st.markdown(f"""
        **Crop**: {crop}  
        **Region**: {region}  
        **Year**: {year}  
        **Reference Temperature**: {temp}°C  
        **Reference Rainfall**: {rainfall}mm  
        **Reference Humidity**: {humidity}%
        
        This illustrative output estimates a yield of **{predicted_yield:.0f} kg/ha** with a 95% confidence interval 
        between {confidence_lower:.0f} and {confidence_upper:.0f} kg/ha based on the TCN-MLP ensemble.
        """)


if __name__ == "__main__":
    render()
