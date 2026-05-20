import streamlit as st
from pathlib import Path
from PIL import Image
from data_loader import apply_global_style


def render():
    apply_global_style()
    
    st.title("📈 Historical Yield Trends")
    
    st.markdown("""
Historical analysis of crop and regional yield patterns from 2000-2024. 
These visualizations show long-term trends and variability in agricultural productivity.
    """)
    
    results_dir = Path(__file__).parent.parent / "results"
    
    st.markdown("---")
    
    # Crop and Regional Trends
    st.subheader("🌾 Crop & Regional Yield Patterns Over Time")
    
    trend_image = results_dir / "historical_yield_trends.png"
    if trend_image.exists():
        img = Image.open(trend_image)
        st.image(img, use_column_width=True, caption="Historical Yield Trends (2000-2024)")
    else:
        st.warning("Historical yield trends image not found. Please run the notebook first.")
    
    st.markdown("""
    ### Key Insights:
    
    **Top Row:**
    - Left: Individual crop yield trajectories over 24 years
    - Right: Regional yield performance patterns
    
    **Bottom Row:**
    - Left: Crop yield trend slopes (green = improving, red = declining)
    - Right: Regional yield trend slopes showing which regions are improving productivity
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Interpretation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Positive Slope (↗️)**
        - Yield productivity is improving
        - Better farm management or tech adoption
        - Climate becoming more favorable
        """)
    
    with col2:
        st.markdown("""
        **Negative Slope (↘️)**
        - Yield productivity declining
        - Climate stress or degradation
        - Need for adaptation strategies
        """)
    
    st.markdown("---")
    
    st.subheader("🔍 Data Summary")
    
    st.info("""
    - **Time Period**: 2000-2024 (24 years of historical data)
    - **Crops Analyzed**: Maize, Rice, Cassava, Yam
    - **Regions Covered**: All 6 geopolitical zones in Nigeria
    - **Data Source**: NASRDA agricultural census + MERRA-2 climate data
    - **Metric**: Average yield (kg/ha) per crop-region combination per year
    """)


if __name__ == "__main__":
    render()
