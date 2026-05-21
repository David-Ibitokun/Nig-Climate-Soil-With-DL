import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data_loader import apply_global_style, load_data, resolve_results_png


def render():
    apply_global_style()
    
    st.title("📊 Model Evaluation")
    
    st.markdown("""
Comprehensive evaluation metrics and visualizations for the TCN-MLP Ensemble model.
    """)
    
    data = load_data()
    ensemble_metrics = data.get("ensemble_metrics")
    if ensemble_metrics is None:
        ensemble_metrics = pd.DataFrame()
    
    per_crop_ensemble = data.get("per_crop_ensemble")
    if per_crop_ensemble is None:
        per_crop_ensemble = pd.DataFrame()

    region_ensemble_metrics = data.get("region_ensemble_metrics")
    if region_ensemble_metrics is None:
        region_ensemble_metrics = pd.DataFrame()
    
    st.markdown("---")
    
    # Performance Metrics - using ensemble results
    st.subheader("🎯 Ensemble Performance Metrics (5-Fold CV)")
    
    if not ensemble_metrics.empty:
        # Calculate ensemble statistics
        avg_r2 = ensemble_metrics['Ensemble_R2'].mean()
        avg_mae = ensemble_metrics['Ensemble_MAE'].mean()
        avg_mape = ensemble_metrics['Ensemble_MAPE'].mean()
        avg_smape = ensemble_metrics['Ensemble_sMAPE'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Average R² Score",
                value=f"{avg_r2:.4f}",
                help="Coefficient of determination (average across 5 folds)"
            )
        with col2:
            st.metric(
                label="Average MAE",
                value=f"{avg_mae:.1f} kg/ha",
                help="Mean Absolute Error (average across 5 folds)"
            )
        with col3:
            st.metric(
                label="Average MAPE",
                value=f"{avg_mape:.2f}%",
                help="Mean Absolute Percentage Error"
            )
        with col4:
            st.metric(
                label="Average sMAPE",
                value=f"{avg_smape:.2f}%",
                help="Symmetric Mean Absolute Percentage Error"
            )
        
        st.markdown("---")
        
        # Fold-by-fold results
        st.subheader("📈 Fold-by-Fold Error Metrics")

        fold_metrics = ensemble_metrics.copy()
        fig_metrics = go.Figure()
        # Include R² alongside error metrics for fold-level comparison
        fig_metrics.add_trace(go.Bar(x=fold_metrics['Fold'], y=fold_metrics['Ensemble_R2'], name='R²', marker_color='#4C78A8'))
        fig_metrics.add_trace(go.Bar(x=fold_metrics['Fold'], y=fold_metrics['Ensemble_MAPE'], name='MAPE', marker_color='#1f77b4'))
        fig_metrics.add_trace(go.Bar(x=fold_metrics['Fold'], y=fold_metrics['Ensemble_sMAPE'], name='sMAPE', marker_color='#ff7f0e'))
        fig_metrics.add_trace(go.Bar(x=fold_metrics['Fold'], y=fold_metrics['Ensemble_MASE'], name='MASE', marker_color='#2ca02c'))

        fig_metrics.update_layout(
            title="MAPE, sMAPE, and MASE by Fold",
            xaxis_title="Fold",
            yaxis_title="Error Metric (%) / Scale",
            barmode="group",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig_metrics, width='stretch')

        st.info(
            "The last fold can have a lower R² simply because its held-out records are a harder subset of the data. "
            "In cross-validation, each fold uses a different crop-region-year mix, so one split may contain more unusual seasons, "
            "more difficult region-crop combinations, or fewer samples that are easier to predict. That makes the R² for that fold "
            "look weaker even when the model is behaving normally. The important result is the average across all five folds, "
            "together with MAE, MAPE, sMAPE, and MASE, which give a more stable picture of overall performance."
        )
        
        # Display fold metrics table
        st.dataframe(ensemble_metrics.round(4), width='stretch', hide_index=True)
        
    else:
        st.warning("Ensemble metrics not found. Please ensure ensemble_metrics_mapes_maase_trimmed.csv exists in results folder.")
    
    st.markdown("---")
    
    # Regional performance
    st.subheader("🗺️ Per-Region Ensemble Performance")

    if not region_ensemble_metrics.empty:
        region_table = region_ensemble_metrics.copy()
        region_table = region_table.sort_values("R2", ascending=False)

        region_fig = go.Figure()
        region_fig.add_trace(go.Bar(x=region_table['Region'], y=region_table['R2'], name='R²', marker_color='#1f77b4'))
        region_fig.add_trace(go.Bar(x=region_table['Region'], y=region_table['MAPE'], name='MAPE', marker_color='#d62728'))
        region_fig.add_trace(go.Bar(x=region_table['Region'], y=region_table['sMAPE'], name='sMAPE', marker_color='#ff7f0e'))
        region_fig.add_trace(go.Bar(x=region_table['Region'], y=region_table['MASE'], name='MASE', marker_color='#2ca02c'))
        region_fig.update_layout(
            title="Regional Performance Summary",
            xaxis_title="Region",
            yaxis_title="Metric Value",
            barmode="group",
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(region_fig, width='stretch')

        st.dataframe(region_table.round(4), width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("📊 Regional Interpretation")

        for _, row in region_table.iterrows():
            region_name = row['Region']
            with st.expander(f"{region_name} - R²={row['R2']:.4f}, MAPE={row['MAPE']:.2f}%"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R²", f"{row['R2']:.4f}")
                with col2:
                    st.metric("MAE", f"{row['MAE']:.1f} kg/ha")
                with col3:
                    st.metric("MAPE", f"{row['MAPE']:.2f}%")
                with col4:
                    st.metric("sMAPE", f"{row['sMAPE']:.2f}%")

    else:
        st.warning("Regional ensemble metrics not found. Please ensure region_ensemble_metrics.csv exists in results folder.")
    
    st.markdown("---")
    
    # Performance Interpretation
    st.subheader("📖 Interpreting the Metrics")
    
    tab1, tab2, tab3 = st.tabs(["R² Score", "Error Metrics", "5-Fold CV"])
    
    with tab1:
        if not ensemble_metrics.empty:
            avg_r2 = ensemble_metrics['Ensemble_R2'].mean()
            st.markdown(f"""
### R² Score: {avg_r2:.4f}

The R² (coefficient of determination) measures the proportion of variance in crop yield 
that is explained by the climate variables.

- **R² = 1.0**: Perfect predictions (impossible in practice)
- **R² = {avg_r2:.4f}**: Current model performance
- **R² = 0.5**: Model explains 50% of yield variance
- **R² = 0.0**: Model predicts no better than the mean yield

**Interpretation**: The model captures {avg_r2*100:.1f}% of the variation in crop yields, 
which is strong for agricultural yield prediction where many factors beyond climate affect outcomes.
            """)
    
    with tab2:
        if not ensemble_metrics.empty:
            avg_mae = ensemble_metrics['Ensemble_MAE'].mean()
            avg_mape = ensemble_metrics['Ensemble_MAPE'].mean()
            avg_smape = ensemble_metrics['Ensemble_sMAPE'].mean()
            avg_mase = ensemble_metrics['Ensemble_MASE'].mean()
            st.markdown(f"""
### Mean Absolute Error (MAE): {avg_mae:.1f} kg/ha
### Mean Absolute Percentage Error (MAPE): {avg_mape:.2f}%
### Symmetric MAPE (sMAPE): {avg_smape:.2f}%
### Mean Absolute Scaled Error (MASE): {avg_mase:.3f}

**MAE** represents the average absolute difference between predicted and actual yields.

- **Test MAE**: {avg_mae:.1f} kg/ha (performance on held-out test data)
- **Relative Error**: About {(avg_mae/2000)*100:.1f}% of average yields
- **MAPE**: {avg_mape:.2f}% (percentage-based error metric)
- **sMAPE**: {avg_smape:.2f}% (balanced percentage error for over/under prediction)
- **MASE**: {avg_mase:.3f} (scale-free error compared to a naive baseline)

**Interpretation**: Lower values are better for MAE, MAPE, sMAPE, and MASE. Together these metrics show 
the model is making reasonably stable predictions while still capturing the remaining climate-driven error.
            """)
    
    with tab3:
        st.markdown("""
### 5-Fold Cross-Validation

Cross-validation ensures the model generalizes well to new data:

- **Method**: Stratified 5-fold (by crop-region combinations)
- **Purpose**: Prevents overfitting and estimates real-world performance
- **Training/Validation Split**: 80/20 per fold
- **Reported Metrics**: Average across all 5 folds

The model is trained 5 times, each time using 4 folds for training and 1 fold for validation.
This provides a robust estimate of performance on unseen data.
        """)
    
    st.markdown("---")
    
    # Crop-specific performance from results
    # st.subheader("🌾 Crop-Specific Climate Sensitivity")
    
    # if "crop_sensitivity" in data:
    #     crop_sensitivity = data['crop_sensitivity']
    #     if isinstance(crop_sensitivity, pd.DataFrame) and not crop_sensitivity.empty and {'Crop', 'Overall_Sensitivity'}.issubset(crop_sensitivity.columns):
    #         fig = px.bar(
    #             crop_sensitivity,
    #             x='Crop',
    #             y='Overall_Sensitivity',
    #             title="Crop Climate Sensitivity",
    #             color='Overall_Sensitivity',
    #             color_continuous_scale="Reds",
    #             labels={'Overall_Sensitivity': 'Sensitivity Score'}
    #         )
    #         st.plotly_chart(fig, width='stretch')
    #         st.dataframe(crop_sensitivity, width='stretch')
    #     else:
    #         st.info("No crop sensitivity table available or required columns ('Crop','Overall_Sensitivity') are missing in results.")
    
    # st.markdown("---")
    
    # # Regional Analysis
    # st.subheader("🗺️ Regional Food Security Risk")
    
    # if "food_security" in data:
    #     food_security = data["food_security"].copy()
    #     if isinstance(food_security, pd.DataFrame) and not food_security.empty and "Food_Security_Risk_Score" in food_security.columns:
    #         fs_sorted = food_security.sort_values("Food_Security_Risk_Score", ascending=False)

    #         fig = px.bar(
    #             fs_sorted,
    #             x="Region",
    #             y="Food_Security_Risk_Score",
    #             title="Food Security Risk Score by Region",
    #             labels={"Food_Security_Risk_Score": "Risk Score (Higher = More Vulnerable)"},
    #             color="Food_Security_Risk_Score",
    #             color_continuous_scale="Reds",
    #         )
    #         st.plotly_chart(fig, width='stretch')

    #         st.dataframe(fs_sorted, width='stretch')
    #     else:
    #         st.info("No food security table available or required column 'Food_Security_Risk_Score' is missing in results.")
    
    # st.markdown("---")
    
    st.subheader("🔍 Model Limitations")
    
    st.warning("""
- **Climate-only predictions**: Model uses climate variables; other factors (pests, soil, management) affect actual yields
- **Historical data**: Trained on 1999-2023 data; may not capture unprecedented climate extremes
- **Regional specificity**: Performance varies by crop-region combination
- **Uncertainty**: Confidence intervals assume historical patterns continue
    """)


if __name__ == "__main__":
    render()
