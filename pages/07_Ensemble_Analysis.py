import streamlit as st
from pathlib import Path
from PIL import Image
from data_loader import apply_global_style


def render():
    apply_global_style()
    
    st.title("⚙️ Ensemble Model Analysis")
    
    st.markdown("""
Detailed analysis of the TCN-MLP ensemble model performance across 5-fold cross-validation.
Visualizes fold-wise results and per-crop model accuracy.
    """)
    
    results_dir = Path(__file__).parent.parent / "results"
    
    st.markdown("---")
    
    # Ensemble dashboard
    st.subheader("🎯 Ensemble Performance Dashboard")
    
    dashboard_image = results_dir / "ensemble_dashboard_highres.png"
    if dashboard_image.exists():
        img = Image.open(dashboard_image)
        st.image(img, use_column_width=True, caption="Fold-wise Ensemble R² | Per-Crop Results | Actual vs Predicted | Regional Uncertainty")
    else:
        st.warning("Ensemble dashboard image not found. Please run the notebook first.")
    
    st.markdown("""
    ### Dashboard Components:
    
    **Top-Left: Fold-wise Ensemble R²**
    - Performance across 5 cross-validation folds
    - Shows model consistency and generalization
    - Variation indicates data stratification quality
    
    **Top-Right: Per-Crop Ensemble R²**
    - R² scores broken down by crop type
    - Identifies which crops are predicted well vs poorly
    - Guides targeted model improvements
    
    **Bottom-Left: Actual vs Predicted Yield**
    - Scatter plot with uncertainty coloring
    - Points close to diagonal = good predictions
    - Color intensity shows ensemble uncertainty
    
    **Bottom-Right: Regional Prediction Uncertainty**
    - Mean standard deviation by region
    - Higher values = less confident predictions
    - Guides data collection priorities
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ensemble Method", "5-Fold CV")
    
    with col2:
        st.metric("Number of Folds", "5")
    
    with col3:
        st.metric("Model Architecture", "TCN-MLP")


if __name__ == "__main__":
    render()
