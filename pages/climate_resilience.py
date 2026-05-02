import streamlit as st
import plotly.express as px

from data_loader import resolve_results_png


def render(data):
    st.title("🛡️ Climate Resilience Analysis")
    
    st.markdown("""
    This page analyzes the resilience of different crop-region combinations against climate stressors.
    Higher resilience index values denote crops/regions better adapted to withstand variability and extremes.
    """)
    
    resilience = data['resilience']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Resilience Index Overview")
        fig = px.box(
            resilience,
            x='Crop',
            y='Resilience_Index',
            color='Crop',
            title="Resilience Index Distribution by Crop",
            labels={'Resilience_Index': 'Resilience Index (0 to 1)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("📊 Stability vs. Resilience")
        fig = px.scatter(
            resilience,
            x='Stability_Score',
            y='Resilience_Index',
            color='Crop',
            size='Baseline_Yield',
            hover_data=['Region'],
            title="Stability vs. Resilience"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Detailed Resilience Data")
    st.dataframe(resilience.sort_values(by="Resilience_Index", ascending=False), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🗺️ Spatial & Seasonal Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Resilience Heatmap", "Climate Impact Analysis", "Seasonal Climate Patterns"])
    
    with tab1:
        try:
            img_path = resolve_results_png("Climate_Resilience_Index_Heatmap.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Climate Resilience Index Heatmap", use_container_width=True)
            else:
                st.warning("Resilience Heatmap image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tab2:
        try:
            img_path = resolve_results_png("Climate_Impact_Analysis.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Climate Impact Analysis", use_container_width=True)
            else:
                st.warning("Climate Impact Analysis image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tab3:
        try:
            img_path = resolve_results_png("Seasonal_Climate_Patterns.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Seasonal Climate Patterns", use_container_width=True)
            else:
                st.warning("Seasonal Climate Patterns image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")


if __name__ == "__main__":
    from data_loader import apply_global_style, load_data

    apply_global_style()
    render(load_data())
