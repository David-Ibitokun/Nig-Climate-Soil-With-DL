import streamlit as st
from pathlib import Path
from PIL import Image
from data_loader import apply_global_style, load_data
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    apply_global_style()
    
    st.title("🌡️ Seasonal Climate Patterns")
    
    st.markdown("""
Analysis of Nigeria's seasonal climate patterns including temperature, rainfall, and humidity.
Understanding these patterns is critical for explaining crop yield variability.
    """)

    st.markdown("""
**Chapter 4 — Results:** Detailed evaluation, SHAP summaries and scenario projection maps that use these climate patterns are available at `notes/CHAPTER_4_TCN-MLP_Results_and_Discussion.md`.
""")
    
    results_dir = Path(__file__).parent.parent / "results"
    
    st.markdown("---")
    
    st.subheader("📅 Seasonal Climate Dynamics")

    # Load processed dataset for plotting monthly seasonal cycles
    data = load_data()
    processed = data.get('processed_dataset') if isinstance(data.get('processed_dataset'), pd.DataFrame) else pd.DataFrame()

    def monthly_mean(frame: pd.DataFrame, feature_code: str):
        vals = []
        months = []
        for m in range(1, 13):
            col = f'{feature_code}_m{m}'
            if col in frame.columns:
                vals.append(float(frame[col].mean()))
            else:
                vals.append(np.nan)
            months.append(m)
        return months, vals

    if not processed.empty:
        months, t_vals = monthly_mean(processed, 'T2M')
        _, r_vals = monthly_mean(processed, 'PRECTOTCORR')
        _, h_vals = monthly_mean(processed, 'RH2M')

        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        # Create three separate figures and place them in Streamlit columns.
        # Streamlit will stack columns vertically on small screens, giving a 3x1 layout automatically.
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=month_names, y=t_vals, mode='lines+markers', line=dict(color='orangered')))
        fig_t.update_layout(title_text='Temperature (T2M)', autosize=True, margin=dict(t=40))
        fig_t.update_yaxes(title_text='°C')

        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=month_names, y=r_vals, marker_color='steelblue'))
        fig_r.update_layout(title_text='Rainfall (PRECTOTCORR)', autosize=True, margin=dict(t=40))
        fig_r.update_yaxes(title_text='mm/day')

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=month_names, y=h_vals, mode='lines+markers', line=dict(color='seagreen')))
        fig_h.update_layout(title_text='Relative Humidity (RH2M)', autosize=True, margin=dict(t=40))
        fig_h.update_yaxes(title_text='%')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig_t, use_container_width=True, config={'responsive': True})
        with col2:
            st.plotly_chart(fig_r, use_container_width=True, config={'responsive': True})
        with col3:
            st.plotly_chart(fig_h, use_container_width=True, config={'responsive': True})
    else:
        climate_patterns = results_dir / "seasonal_climate_patterns.png"
        if climate_patterns.exists():
            img = Image.open(climate_patterns)
            st.image(img, width='stretch', caption='Monthly Climate Patterns (Temperature, Rainfall, Humidity)')
        else:
            st.warning("Seasonal climate patterns data not found. Please run the notebook first.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📈 Annual Climate Patterns Over Time")

    # Prefer generating the plot directly from processed dataset for higher quality
    data = load_data()
    processed = data.get('processed_dataset') if isinstance(data.get('processed_dataset'), pd.DataFrame) else pd.DataFrame()

    def annual_mean_feature(frame: pd.DataFrame, feature_code: str) -> pd.DataFrame:
        month_cols = [f"{feature_code}_m{m}" for m in range(1, 13) if f"{feature_code}_m{m}" in frame.columns]
        if not month_cols:
            return pd.DataFrame()
        tmp = frame.copy()
        tmp['__feat_mean__'] = tmp[month_cols].mean(axis=1)
        annual = tmp.groupby('Year')['__feat_mean__'].mean().reset_index()
        return annual.dropna()

    if not processed.empty:
        regions = sorted(processed['Region'].dropna().unique())
        region_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        region_colors = {region: region_palette[i % len(region_palette)] for i, region in enumerate(regions)}
        month_names = None

        def annual_feature_by_region(feature_code: str):
            fig = go.Figure()
            feature_labels = {
                'PRECTOTCORR': ('Precipitation (mm/day)', 'steelblue'),
                'T2M': ('Temperature (°C)', 'orangered'),
                'RH2M': ('Relative Humidity (%)', 'seagreen'),
            }
            y_label, color = feature_labels[feature_code]

            for region in regions:
                region_df = processed[processed['Region'] == region]
                annual_df = annual_mean_feature(region_df, feature_code)
                if annual_df.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=annual_df['Year'],
                        y=annual_df['__feat_mean__'],
                        mode='lines+markers',
                        name=region,
                        line=dict(color=region_colors[region]),
                        marker=dict(color=region_colors[region]),
                    )
                )

            fig.update_layout(
                title=dict(
                    text=f'Annual {y_label} Trends by Region',
                    x=0.5,
                    xanchor='center',
                    y=0.96,
                    yanchor='top',
                    font=dict(size=14),
                ),
                autosize=True,
                height=430,
                margin=dict(t=95, b=105, l=55, r=25),
                legend=dict(
                    orientation='h',
                    yanchor='top',
                    y=-0.28,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=9),
                ),
            )
            fig.update_xaxes(title_text='Year')
            fig.update_yaxes(title_text=y_label)
            return fig
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(annual_feature_by_region('PRECTOTCORR'), use_container_width=True, config={'responsive': True})
        with col2:
            st.plotly_chart(annual_feature_by_region('T2M'), use_container_width=True, config={'responsive': True})
        with col3:
            st.plotly_chart(annual_feature_by_region('RH2M'), use_container_width=True, config={'responsive': True})
    else:
        # fallback to image if processed dataset not available
        annual_trends = results_dir / "climate_trends_by_region_1999_2023.png"
        if annual_trends.exists():
            from PIL import Image
            annual_img = Image.open(annual_trends)
            st.image(annual_img, width='stretch', caption='Climate trends by region (1999-2023)')
        else:
            st.warning("Annual climate trends data not found. Run the evaluation notebook to generate results or provide processed_dataset.")

    # st.markdown("""
    # ### Annual Climate Parameters Used in Training

    # - `T2M` - Mean temperature at 2m
    # - `T2M_MAX` - Max temperature at 2m
    # - `T2M_MIN` - Min temperature at 2m
    # - `TS` - Land surface temperature
    # - `PRECTOTCORR` - Bias-corrected total precipitation
    # - `RH2M` - Relative humidity at 2m
    # - `QV2M` - Specific humidity at 2m
    # - `T2MDEW` - Dew point temperature at 2m
    # - `T2MWET` - Wet bulb temperature at 2m
    # """)

    st.markdown("""
    ### What Each Climate Parameter Means

    | Parameter | Meaning | What it tells us |
    |---|---|---|
    | `T2M` | Mean air temperature at 2 meters | General warmth of the growing environment; affects crop development speed. |
    | `T2M_MAX` | Maximum air temperature at 2 meters | Heat stress during hot periods; can reduce pollination and grain filling. |
    | `T2M_MIN` | Minimum air temperature at 2 meters | Night-time cooling; influences respiration and recovery from daytime heat. |
    | `TS` | Land surface temperature | Surface heating of soil and canopy; useful for drying stress and heat exposure. |
    | `PRECTOTCORR` | Bias-corrected total precipitation | Rainfall supply; drives soil moisture and supports rainfed cropping. |
    | `RH2M` | Relative humidity at 2 meters | Moisture in the air; affects evapotranspiration and disease pressure. |
    | `QV2M` | Specific humidity at 2 meters | Actual water vapour content in air; another measure of atmospheric moisture. |
    | `T2MDEW` | Dew point temperature at 2 meters | Temperature at which moisture condenses; higher values mean moister air. |
    | `T2MWET` | Wet bulb temperature at 2 meters | Combined heat-and-moisture stress indicator; important for crop comfort and evaporation. |
    """)

    st.markdown("""
    ### Why These Variables Matter for Yield

    These nine variables capture the main weather conditions the model learns from: heat, cold, rainfall, and air moisture.
    Together they describe whether crops are likely to experience enough water, too much heat, or stressful dry conditions during the season.
    """)
    
    st.markdown("""
    ### Climate Pattern Analysis:
    
    **Left: Temperature (T2M)**
    - Monthly mean temperature variations
    - Peak warmth typically June-September
    - Critical for crop growth phases
    
    **Center: Rainfall**
    - Monthly precipitation patterns
    - Monsoon season (May-September) shows peak rainfall
    - Crucial for irrigation requirements
    
    **Right: Relative Humidity**
    - Monthly humidity patterns
    - Higher during rainy season
    - Affects disease pressure and evapotranspiration
    """)
    
    st.markdown("---")
    
    st.subheader("🔄 Seasonal Cycle Interpretation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Pre-Monsoon (Jan-Apr)**
        - Temp: Rising ↗️
        - Rain: Minimal
        - Status: Dry season
        - Impact: Requires irrigation
        """)
    
    with col2:
        st.markdown("""
        **Monsoon (May-Sep)**
        - Temp: Peak highs
        - Rain: Maximum ↗️
        - Status: Wet season
        - Impact: Good for rainfed crops
        """)
    
    with col3:
        st.markdown("""
        **Post-Monsoon (Oct-Dec)**
        - Temp: Cooling ↘️
        - Rain: Declining
        - Status: Transitional
        - Impact: End of season harvest
        """)
    
    st.markdown("---")
    
    st.info("""
    **Climate-Yield Connection**: Seasonal climate patterns directly influence:
    - Planting decisions
    - Water stress levels
    - Growing season length
    - Pest and disease pressure
    - Final yield outcomes
    """)


if __name__ == "__main__":
    render()
