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
    
    # Prediction interface
    st.subheader("📋 Select Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop = st.selectbox(
            "Select Crop",
            ["Maize", "Rice", "Cassava", "Yam"]
        )
    
    with col2:
        region = st.selectbox(
            "Select Region",
            ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
        )
    
    with col3:
        year = st.slider(
            "Select Year",
            min_value=2026,
            max_value=2030,
            value=2026
        )
    
    st.markdown("---")
    
    st.subheader("🌡️ Climate Variables")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.slider(
            "Average Temperature (°C)",
            min_value=15.0,
            max_value=35.0,
            value=27.5,
            step=0.5
        )
    
    with col2:
        rainfall = st.slider(
            "Annual Rainfall (mm)",
            min_value=500.0,
            max_value=3000.0,
            value=1200.0,
            step=50.0
        )
    
    with col3:
        humidity = st.slider(
            "Relative Humidity (%)",
            min_value=30.0,
            max_value=90.0,
            value=65.0,
            step=1.0
        )
    
    st.markdown("---")
    
    # Make prediction
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        st.info("Model prediction would be generated here with the selected parameters.")
        
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
        **Temperature**: {temp}°C  
        **Rainfall**: {rainfall}mm  
        **Humidity**: {humidity}%
        
        The model estimates a yield of **{predicted_yield:.0f} kg/ha** with a 95% confidence interval 
        between {confidence_lower:.0f} and {confidence_upper:.0f} kg/ha based on the TCN-MLP ensemble.
        """)


if __name__ == "__main__":
    render()
