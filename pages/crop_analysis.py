import streamlit as st
import plotly.express as px

from data_loader import resolve_results_png


def render(data):
    st.title("🌾 Crop Performance & Sensitivity Analysis")
    
    crop_sensitivity = data['crop_sensitivity']
    
    st.subheader("📊 Crop Climate Sensitivity")
    st.markdown("""
    This table shows the sensitivity of each crop to different climate stressors. 
    Higher positive values indicate a stronger sensitivity (yield change per unit of stress).
    """)
    st.dataframe(crop_sensitivity, use_container_width=True)
    
    # Visualization of sensitivity
    fig = px.bar(
        crop_sensitivity,
        x='Crop',
        y='Overall_Sensitivity',
        title="Overall Climate Sensitivity by Crop",
        color='Overall_Sensitivity',
        color_continuous_scale="Reds",
        labels={'Overall_Sensitivity': 'Overall Sensitivity Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Crop Specific Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Crop Parity", "Feature Correlations", "Historical Trends"])
    
    with tab1:
        try:
            img_path = resolve_results_png("Crop_Specific_Parity.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Crop Specific Parity Plots", use_container_width=True)
            else:
                st.warning("Crop Specific Parity image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tab2:
        try:
            img_path = resolve_results_png("Feature_Correlation_with_Yield.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Feature Correlation with Yield", use_container_width=True)
            else:
                st.warning("Feature Correlation image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tab3:
        try:
            img_path = resolve_results_png("Historical_Yield_Trends.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Historical Yield Trends", use_container_width=True)
            else:
                st.warning("Historical Yield Trends image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")


if __name__ == "__main__":
    from data_loader import apply_global_style, load_data

    apply_global_style()
    render(load_data())
