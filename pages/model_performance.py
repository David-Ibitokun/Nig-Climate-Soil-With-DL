import streamlit as st

from data_loader import resolve_results_png


def render(data):
    st.title("📊 TCN-MLP Model Performance")
    
    # Architecture overview
    st.markdown("""
    <div style="padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin-bottom: 2rem;">
        <h4 style="margin-top: 0; color: #0066cc;">Hybrid Architecture Overview (4 Crops)</h4>
        <p style="margin-bottom: 0.5rem;"><strong>Temporal Convolutional Network (TCN)</strong>: Captures temporal dependencies and seasonal patterns in climate data</p>
        <p style="margin-bottom: 0;"><strong>Multi-Layer Perceptron (MLP)</strong>: Processes derived features and learns complex feature interactions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Performance Metrics
    st.subheader("📊 Key Performance Metrics")
    
    meta = data.get('metadata', {})
    perf = meta.get("performance", {}) if isinstance(meta, dict) else {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Test R² Score",
            value=f"{perf.get('final_test_r2', 0):.4f}"
        )
    with col2:
        st.metric(
            label="Test MAE",
            value=f"{perf.get('final_test_mae', 0):.1f} kg/ha"
        )
    with col3:
        st.metric(
            label="Validation MAE (Baseline)",
            value=f"{perf.get('cv_val_mae_mean', 0):.1f} kg/ha"
        )
    with col4:
        st.metric(
            label="Features",
            value=f"{(meta.get('data', {}) or {}).get('n_temporal_features', 0)} Temporal",
            help=", ".join((meta.get('features', {}) or {}).get('unique_features', []))
        )
    
    st.markdown("---")
    
    # Performance Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance Graphs", "📋 Evaluation Summary", "🏗️ Architecture"])
    
    with tab1:
        st.subheader("Model Performance Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                img_path = resolve_results_png("TCN_MLP_4Crops_Performance.png")
                if img_path.exists():
                    st.image(str(img_path), caption="Overall Model Performance", use_container_width=True)
                else:
                    st.warning("Performance image not found.")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                
        with col2:
            try:
                img_path = resolve_results_png("Train_Val_Test_Comparison.png")
                if img_path.exists():
                    st.image(str(img_path), caption="Train vs Validation vs Test", use_container_width=True)
                else:
                    st.warning("Comparison image not found.")
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    with tab2:
        st.subheader("📋 Complete Evaluation Summary")
        summary_text = data.get('summary', 'No summary available.')
        st.text(summary_text)
        
    with tab3:
        st.subheader("🏗️ Model Details")
        st.json(meta)
        
        st.markdown("""
        ### Architecture Rationale
        - **TCN**: Temporal convolutions are ideal for capturing seasonal climate patterns
        - **MLP integration**: Combines temporal patterns with categorical features (Crop, Region)
        - **Huber Loss**: Robust to outliers in the yield data
        - **MultiHeadAttention**: Captures complex dependencies across the sequence
        """)


if __name__ == "__main__":
    from data_loader import apply_global_style, load_data

    apply_global_style()
    render(load_data())
