import streamlit as st
from pathlib import Path
from PIL import Image
from data_loader import apply_global_style


@st.cache_data(show_spinner=False)
def _load_image(path: Path):
    if not path.exists():
        return None
    with Image.open(path) as img:
        return img.copy()


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
    img = _load_image(resilience_image)
    if img is not None:
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

    **How this image was generated (summary):**
    1. The processed dataset of historical yields (`data/processed_dataset.csv`) was grouped by `Crop` and `Region` to build yield time series for each cell.
    2. For each crop-region combination we computed the coefficient of variation (CV = std / mean) of historical yields.
    3. The resilience index was derived as `1 / (1 + CV)`, producing values in (0,1] where higher means more stable yields.
    4. Values were clipped and optionally smoothed to avoid outliers dominating the color scale, then pivoted into a matrix (rows=crops, cols=regions).
    5. A heatmap was created (matplotlib / seaborn) with a diverging colormap and annotations and saved to `results/climate_resilience_index_heatmap.png`.
    6. The notebook also saves the underlying numeric CSV so results can be inspected programmatically.

    This representation highlights where yields are historically more stable (green) versus more variable (red), helping prioritize resilience interventions.
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
