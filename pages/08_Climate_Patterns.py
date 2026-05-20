import streamlit as st
from pathlib import Path
from PIL import Image
from data_loader import apply_global_style


def render():
    apply_global_style()
    
    st.title("🌡️ Seasonal Climate Patterns")
    
    st.markdown("""
Analysis of Nigeria's seasonal climate patterns including temperature, rainfall, and humidity.
Understanding these patterns is critical for explaining crop yield variability.
    """)
    
    results_dir = Path(__file__).parent.parent / "results"
    
    st.markdown("---")
    
    st.subheader("📅 Seasonal Climate Dynamics")
    
    climate_patterns = results_dir / "seasonal_climate_patterns.png"
    if climate_patterns.exists():
        img = Image.open(climate_patterns)
        st.image(img, width="stretch", caption="Monthly Climate Patterns (Temperature, Rainfall, Humidity)")
    else:
        st.warning("Seasonal climate patterns image not found. Please run the notebook first.")
    
    st.markdown("""
    ### Climate Pattern Analysis:
    
    **Left: Temperature (T2M)**
    - Monthly mean temperature variations
    - Peak warmth typically June-September
    - Critical for crop growth phases
    
    **Center: Rainfall**
    - Monthly precipitation patterns
    - Monsoon season (May-September) shows peak rainfall
    - Crucial for irrigation requirements
    
    **Right: Relative Humidity**
    - Monthly humidity patterns
    - Higher during rainy season
    - Affects disease pressure and evapotranspiration
    """)
    
    st.markdown("---")
    
    st.subheader("🔄 Seasonal Cycle Interpretation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Pre-Monsoon (Jan-Apr)**
        - Temp: Rising ↗️
        - Rain: Minimal
        - Status: Dry season
        - Impact: Requires irrigation
        """)
    
    with col2:
        st.markdown("""
        **Monsoon (May-Sep)**
        - Temp: Peak highs
        - Rain: Maximum ↗️
        - Status: Wet season
        - Impact: Good for rainfed crops
        """)
    
    with col3:
        st.markdown("""
        **Post-Monsoon (Oct-Dec)**
        - Temp: Cooling ↘️
        - Rain: Declining
        - Status: Transitional
        - Impact: End of season harvest
        """)
    
    st.markdown("---")
    
    st.info("""
    **Climate-Yield Connection**: Seasonal climate patterns directly influence:
    - Planting decisions
    - Water stress levels
    - Growing season length
    - Pest and disease pressure
    - Final yield outcomes
    """)


if __name__ == "__main__":
    render()
