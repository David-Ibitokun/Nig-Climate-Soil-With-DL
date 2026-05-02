import streamlit as st
import pandas as pd

from data_loader import resolve_results_png


def render(data):
    st.title("💡 Adaptive Strategies & Recommendations")
    
    st.markdown("""
    Based on the climate vulnerability models and SHAP feature importance analysis, 
    the following adaptive strategies are recommended for policymakers and stakeholders.
    """)
    
    strategies = data['strategies']
    
    st.subheader("📋 Key Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Style the dataframe for better display
        st.dataframe(strategies, use_container_width=True)
        
    with col2:
        st.info("""
        **Intervention Types:**
        - **Regional Focus**: Actions targeted at specific geopolitical zones facing extreme risks.
        - **Crop-Specific**: Tailored interventions for sensitive crops (e.g., drought-resistant seeds).
        """)
        
    st.markdown("---")
    
    st.subheader("🛠️ Strategy Effectiveness & Drivers")
    
    tab1, tab2 = st.tabs(["Intervention Effectiveness", "Key Climate Drivers (SHAP)"])
    
    with tab1:
        try:
            img_path = resolve_results_png("Adaptation_Intervention_Effectiveness.png")
            if img_path.exists():
                 st.image(str(img_path), caption="Simulated Effectiveness of Various Interventions", use_container_width=True)
            else:
                st.warning("Adaptation Intervention Effectiveness image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tab2:
        try:
            img_path = resolve_results_png("SHAP_Feature_Importance.png")
            if img_path.exists():
                 st.image(str(img_path), caption="SHAP Feature Importance Analysis", use_container_width=True)
                st.markdown(
                    """
                    *SHAP values explain which climate and temporal features have the highest impact on yield predictions.*
                    """
                )
            else:
                st.warning("SHAP Feature Importance image not found.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    from data_loader import apply_global_style, load_data

    apply_global_style()
    render(load_data())