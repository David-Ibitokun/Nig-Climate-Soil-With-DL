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
        st.subheader("📈 Fold-by-Fold Results")
        
        fig_metrics = go.Figure(data=[
            go.Bar(x=ensemble_metrics['Fold'], y=ensemble_metrics['Ensemble_R2'], name='R² Score', marker_color='#1f77b4'),
        ])
        
        fig_metrics.add_trace(
            go.Scatter(x=ensemble_metrics['Fold'], y=ensemble_metrics['Ensemble_MAE']/1000, 
                      name='MAE (÷1000)', mode='lines+markers', marker_color='#ff7f0e', yaxis='y2')
        )
        
        fig_metrics.update_layout(
            title="Ensemble Metrics by Fold",
            xaxis_title="Fold",
            yaxis_title="R² Score",
            yaxis2=dict(title="MAE (kg/ha÷1000)", overlaying="y", side="right"),
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Display fold metrics table
        st.dataframe(ensemble_metrics.round(4), use_container_width=True, hide_index=True)
        
    else:
        st.warning("Ensemble metrics not found. Please ensure ensemble_metrics_mapes_maase_trimmed.csv exists in results folder.")
    
    st.markdown("---")
    
    # Crop-specific performance
    st.subheader("🌾 Per-Crop Ensemble Performance")
    
    if not per_crop_ensemble.empty:
        # Create visualization
        fig_crops = go.Figure()
        
        fig_crops.add_trace(go.Bar(
            x=per_crop_ensemble['Crop'],
            y=per_crop_ensemble['Ens_R2'],
            name='R² Score',
            marker_color='#1f77b4'
        ))
        
        fig_crops.update_layout(
            title="Ensemble R² Score by Crop",
            xaxis_title="Crop",
            yaxis_title="R² Score",
            height=400
        )
        st.plotly_chart(fig_crops, use_container_width=True)
        
        # Crop metrics table
        st.dataframe(per_crop_ensemble.round(4), use_container_width=True, hide_index=True)
        
        # Per-crop insights
        st.markdown("---")
        st.subheader("📊 Per-Crop Performance Insights")
        
        for _, row in per_crop_ensemble.iterrows():
            crop_name = row['Crop']
            r2 = row['Ens_R2']
            mae = row['Ens_MAE']
            mape = row['Ens_MAPE']
            
            with st.expander(f"{crop_name} - R²={r2:.4f}, MAE={mae:.1f} kg/ha"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("MAE", f"{mae:.1f} kg/ha")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col4:
                    st.metric("sMAPE", f"{row['Ens_sMAPE']:.2f}%")
    else:
        st.warning("Per-crop ensemble metrics not found. Please ensure per_crop_ensemble_mapes_maase_trimmed.csv exists in results folder.")
    
    st.markdown("---")
    
    # Performance Interpretation
    st.subheader("📖 Interpreting the Metrics")
    
    tab1, tab2, tab3 = st.tabs(["R² Score", "MAE/MAPE", "5-Fold CV"])
    
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
            st.markdown(f"""
### Mean Absolute Error (MAE): {avg_mae:.1f} kg/ha
### Mean Absolute Percentage Error (MAPE): {avg_mape:.2f}%

**MAE** represents the average absolute difference between predicted and actual yields.

- **Test MAE**: {avg_mae:.1f} kg/ha (performance on held-out test data)
- **Relative Error**: About {(avg_mae/2000)*100:.1f}% of average yields
- **MAPE**: {avg_mape:.2f}% (percentage-based error metric)

**Interpretation**: On average, predictions are off by {avg_mae:.1f} kg/ha ({avg_mape:.2f}%), 
which represents good accuracy for climate-based yield forecasting.
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
    st.subheader("🌾 Crop-Specific Climate Sensitivity")
    
    if "crop_sensitivity" in data:
        crop_sensitivity = data['crop_sensitivity']
        
        fig = px.bar(
            crop_sensitivity,
            x='Crop',
            y='Overall_Sensitivity',
            title="Crop Climate Sensitivity",
            color='Overall_Sensitivity',
            color_continuous_scale="Reds",
            labels={'Overall_Sensitivity': 'Sensitivity Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(crop_sensitivity, use_container_width=True)
    
    st.markdown("---")
    
    # Regional Analysis
    st.subheader("🗺️ Regional Food Security Risk")
    
    if "food_security" in data:
        food_security = data["food_security"].copy()
        fs_sorted = food_security.sort_values("Food_Security_Risk_Score", ascending=False)
        
        fig = px.bar(
            fs_sorted,
            x="Region",
            y="Food_Security_Risk_Score",
            title="Food Security Risk Score by Region",
            labels={"Food_Security_Risk_Score": "Risk Score (Higher = More Vulnerable)"},
            color="Food_Security_Risk_Score",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(fs_sorted, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔍 Model Limitations")
    
    st.warning("""
- **Climate-only predictions**: Model uses climate variables; other factors (pests, soil, management) affect actual yields
- **Historical data**: Trained on 2000-2024 data; may not capture unprecedented climate extremes
- **Regional specificity**: Performance varies by crop-region combination
- **Uncertainty**: Confidence intervals assume historical patterns continue
    """)


if __name__ == "__main__":
    render()
