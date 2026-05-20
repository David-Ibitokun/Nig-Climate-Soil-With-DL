import streamlit as st
from pathlib import Path
from PIL import Image
from data_loader import apply_global_style


def render():
    apply_global_style()
    
    st.title("💪 Climate Resilience Index")
    
    st.markdown("""
Climate resilience measures how well each crop-region combination can withstand climate variability.
Higher resilience indicates more stable yields despite climate fluctuations.
    """)
    
    results_dir = Path(__file__).parent.parent / "results"
    
    st.markdown("---")
    
    st.subheader("🗺️ Resilience by Crop & Region Heatmap")
    
    resilience_image = results_dir / "climate_resilience_index_heatmap.png"
    if resilience_image.exists():
        img = Image.open(resilience_image)
        st.image(img, width="stretch", caption="Climate Resilience Index: Green = High Resilience, Red = Low Resilience")
    else:
        st.warning("Climate resilience index image not found. Please run the notebook first.")
    
    st.markdown("""
    ### Interpreting the Heatmap:
    
    **Color Coding:**
    - **🟩 Green (0.7-1.0)**: High resilience - stable yields, low climate sensitivity
    - **🟨 Yellow (0.4-0.7)**: Moderate resilience - some yield variability
    - **🟥 Red (0.0-0.4)**: Low resilience - high yield variability, vulnerable
    
    **Resilience Index Formula:**
    - Based on coefficient of variation (CV) in historical yields
    - CV = Standard Deviation / Mean Yield
    - Index = 1 / (1 + CV) - measures yield stability
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Resilience Factors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Crop Resilience**
        
        Varies by:
        - Root depth
        - Growing season
        - Water needs
        - Market demand
        """)
    
    with col2:
        st.markdown("""
        **Regional Resilience**
        
        Affected by:
        - Climate patterns
        - Soil quality
        - Infrastructure
        - Farm tech adoption
        """)
    
    with col3:
        st.markdown("""
        **Interaction Effects**
        
        Crop-Region match:
        - Maize needs good rainfall
        - Cassava tolerates drought
        - Rice needs irrigation
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 Resilience Enhancement Strategies")
    
    strategies = {
        "🌾 Crop Selection": "Plant climate-resilient varieties (green zones)",
        "💧 Irrigation": "Supplement rainfall in low-resilience areas",
        "🔄 Crop Rotation": "Diversify crops to spread risk",
        "🧬 Breeding": "Develop drought/heat-tolerant cultivars",
        "📚 Knowledge": "Train farmers on climate-smart practices",
        "📍 Market Access": "Connect to value chains for better prices"
    }
    
    for strategy, description in strategies.items():
        st.write(f"**{strategy}**: {description}")


if __name__ == "__main__":
    render()
