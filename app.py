import streamlit as st
from data_loader import apply_global_style, load_data


def main():
    apply_global_style()

    data = load_data()
    meta = data.get("metadata") or {}
    perf = meta.get("performance", {}) if isinstance(meta, dict) else {}

    st.title("🌾 Enhanced Crop Yield Prediction")
    st.markdown(
        """
**Predicting crop yield from climate sequences** using a TCN–MLP Ensemble architecture.

Navigate through the sidebar to explore predictions, data, model architecture, evaluation metrics, and project details.
"""
    )

    st.markdown("---")

    # Overview metrics
    st.subheader("📊 Model Overview")
    col1, col2, col3, col4 = st.columns(4)

    # Prefer ensemble prediction summary when available in metadata.performance
    ensemble_mean = perf.get("ensemble_mean")
    ensemble_std = perf.get("ensemble_std")
    ensemble_lower = perf.get("ensemble_lower")
    ensemble_upper = perf.get("ensemble_upper")

    with col1:
        if ensemble_mean is not None:
            st.metric("Ensemble Mean Prediction", f"{ensemble_mean:.0f} kg/ha")
        else:
            st.metric("Test R²", f"{perf.get('final_test_r2', 0):.4f}")

    with col2:
        if ensemble_std is not None:
            st.metric("Ensemble Std Dev", f"{ensemble_std:.0f} kg/ha")
        elif ensemble_lower is not None and ensemble_upper is not None:
            st.metric("Ensemble CI", f"{ensemble_lower:.0f}–{ensemble_upper:.0f} kg/ha")
        else:
            st.metric("Test MAE", f"{perf.get('final_test_mae', 0):.1f} kg/ha")

    with col3:
        st.metric("Crops", "4", "Maize, Rice, Cassava, Yam")

    with col4:
        st.metric(
            "Temporal Features",
            f"{(meta.get('data', {}) or {}).get('n_temporal_features', 0)}",
        )

    st.markdown("---")

    # Quick start guide
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
#### **Make Prediction**
Use the trained ensemble model to predict crop yields based on climate parameters for different regions and years.
        """)
    
    with col2:
        st.markdown("""
#### **Data Explorer**
Explore the climate and yield datasets, distributions, and relationships across crops and regions.
        """)
    
    with col3:
        st.markdown("""
#### **Model Architecture**
Learn about the TCN-MLP hybrid architecture and how it captures temporal patterns.
        """)

    st.markdown("---")

    st.subheader("📚 Project Information")
    st.markdown("""
- **Objective**: Predict crop yields from climate sequences using deep learning
- **Scope**: 4 crops (Maize, Rice, Cassava, Yam) across Nigeria's geopolitical zones
- **Model**: TCN-MLP Ensemble with 5-fold cross-validation
- **Data**: Climate sequences with yield observations

Use the navigation menu to explore different sections of the application.
    """)


if __name__ == "__main__":
    main()
