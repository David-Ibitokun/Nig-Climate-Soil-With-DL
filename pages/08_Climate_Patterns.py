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

        fig = make_subplots(rows=1, cols=3, subplot_titles=("Temperature (T2M)", "Rainfall (PRECTOTCORR)", "Relative Humidity (RH2M)"))

        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        fig.add_trace(go.Scatter(x=month_names, y=t_vals, mode='lines+markers', line=dict(color='orangered'), name='Temperature (T2M)'), row=1, col=1)
        fig.update_yaxes(title_text='°C', row=1, col=1)

        fig.add_trace(go.Bar(x=month_names, y=r_vals, marker_color='steelblue', name='Rainfall (PRECTOTCORR)'), row=1, col=2)
        fig.update_yaxes(title_text='mm/day', row=1, col=2)

        fig.add_trace(go.Scatter(x=month_names, y=h_vals, mode='lines+markers', line=dict(color='seagreen'), name='Relative Humidity (RH2M)'), row=1, col=3)
        fig.update_yaxes(title_text='%', row=1, col=3)

        fig.update_layout(
            height=420,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.18, xanchor='center', x=0.5),
            margin=dict(t=120, b=20),
        )
        st.plotly_chart(fig, width='stretch')
    else:
        climate_patterns = results_dir / "seasonal_climate_patterns.png"
        if climate_patterns.exists():
            img = Image.open(climate_patterns)
            st.image(img, width='stretch', caption='Monthly Climate Patterns (Temperature, Rainfall, Humidity)')
        else:
            st.warning("Seasonal climate patterns data not found. Please run the notebook first.")

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
        n_regions = len(regions)
        cols = 3
        rows = (n_regions + cols - 1) // cols

        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, subplot_titles=regions, specs=[[{"secondary_y": True}]*cols for _ in range(rows)])

        for i, region in enumerate(regions):
            r = i // cols + 1
            c = i % cols + 1

            region_df = processed[processed['Region'] == region]
            p_df = annual_mean_feature(region_df, 'PRECTOTCORR')
            t_df = annual_mean_feature(region_df, 'T2M')
            h_df = annual_mean_feature(region_df, 'RH2M')

            if not p_df.empty:
                fig.add_trace(
                    go.Scatter(x=p_df['Year'], y=p_df['__feat_mean__'], mode='lines+markers', name=f'{region} Precip (mm/day)', line=dict(color='steelblue')),
                    row=r, col=c, secondary_y=False
                )
            if not t_df.empty:
                fig.add_trace(
                    go.Scatter(x=t_df['Year'], y=t_df['__feat_mean__'], mode='lines+markers', name=f'{region} Temp (°C)', line=dict(color='orangered')),
                    row=r, col=c, secondary_y=True
                )
            if not h_df.empty:
                fig.add_trace(
                    go.Scatter(x=h_df['Year'], y=h_df['__feat_mean__'], mode='lines+markers', name=f'{region} Humidity (%)', line=dict(color='seagreen')),
                    row=r, col=c, secondary_y=True
                )

            fig.update_xaxes(title_text='Year', row=r, col=c)

        fig.update_layout(height=300 * rows, showlegend=False, title_text='Climate trends by region (1999-2023)')
        st.plotly_chart(fig, width='stretch')
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
