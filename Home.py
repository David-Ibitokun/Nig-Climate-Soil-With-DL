import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_loader import apply_global_style, load_data
import re

CLIMATE_PARAMETER_CODES = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "TS",
    "PRECTOTCORR",
    "RH2M",
    "QV2M",
    "T2MDEW",
    "T2MWET",
]


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

    # --- Model evaluation summary (compact) ---
    st.subheader("📈 Model Evaluation Summary")

    data = load_data()
    ensemble_metrics = data.get("ensemble_metrics") if isinstance(data.get("ensemble_metrics"), pd.DataFrame) else None
    per_crop_ensemble = data.get("per_crop_ensemble") if isinstance(data.get("per_crop_ensemble"), pd.DataFrame) else None

    avg_r2 = None
    avg_mae = None
    if ensemble_metrics is not None and not ensemble_metrics.empty:
        avg_r2 = ensemble_metrics['Ensemble_R2'].mean()
        avg_mae = ensemble_metrics['Ensemble_MAE'].mean()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Avg R² (5-fold)", f"{avg_r2:.3f}")
            st.metric("Avg MAE", f"{avg_mae:.1f} kg/ha")

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ensemble_metrics['Fold'], y=ensemble_metrics['Ensemble_R2'], name='R²', marker_color='#1f77b4'))
            fig.update_layout(title='Ensemble R² by Fold', xaxis_title='Fold', yaxis_title='R²', height=260, margin=dict(t=40,b=20))
            st.plotly_chart(fig, width='stretch')

        # show a compact per-crop table
        if per_crop_ensemble is not None and not per_crop_ensemble.empty:
            st.markdown("**Per-crop summary (top)**")
            st.dataframe(per_crop_ensemble[['Crop','Ens_R2','Ens_MAE']].sort_values('Ens_R2', ascending=False).round(3).reset_index(drop=True), width='stretch')
    else:
        st.info("Model evaluation metrics not available. Open the 'Model Evaluation' page for full details.")

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
        if avg_r2 is not None:
            st.metric("Avg R² (5-fold)", f"{avg_r2:.3f}")
        elif ensemble_mean is not None:
            st.metric("Ensemble Mean Prediction", f"{ensemble_mean:.0f} kg/ha")
        else:
            st.metric("Avg R² (5-fold)", "N/A")

    with col2:
        if avg_mae is not None:
            st.metric("Avg MAE", f"{avg_mae:.1f} kg/ha")
        elif ensemble_std is not None:
            st.metric("Ensemble Std Dev", f"{ensemble_std:.0f} kg/ha")
        elif ensemble_lower is not None and ensemble_upper is not None:
            st.metric("Ensemble CI", f"{ensemble_lower:.0f}–{ensemble_upper:.0f} kg/ha")
        else:
            st.metric("Avg MAE", "N/A")

    with col3:
        st.metric("Crops", "4", "Maize, Rice, Cassava, Yam")

    with col4:
        # Prefer metadata value, otherwise infer from processed_dataset column patterns like 'T2M_m1'..'T2M_m12'
        n_temporal = None
        meta_data = meta.get('data', {}) if isinstance(meta, dict) else {}
        if isinstance(meta_data, dict) and meta_data.get('n_temporal_features'):
            try:
                n_temporal = int(meta_data.get('n_temporal_features'))
            except Exception:
                n_temporal = None

        if n_temporal is None:
            # Use the exact nine climate parameters that were used in training.
            n_temporal = len(CLIMATE_PARAMETER_CODES)

        if n_temporal is None:
            # Fallback: infer from processed_dataset column patterns like 'T2M_m1'..
            processed = data.get('processed_dataset') if isinstance(data.get('processed_dataset'), pd.DataFrame) else pd.DataFrame()
            if not processed.empty:
                prefixes = set()
                for c in processed.columns:
                    m = re.match(r"(.+)_m1$", c)
                    if m:
                        prefixes.add(m.group(1))
                n_temporal = len(prefixes)
            else:
                n_temporal = 0

        st.metric(
            "Temporal Features",
            f"{n_temporal}",
            help="9 climate parameters used in training: " + ", ".join(CLIMATE_PARAMETER_CODES),
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
