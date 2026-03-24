"""
Streamlit App: Climate Change & Food Security Impact Analysis in Nigeria
Analyzing the impact of climate change and soil conditions on food security using TCN-MLP models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Nigeria Food Security Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        background-color: #ffebee;
        border-left-color: #d32f2f;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left-color: #f57c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left-color: #388e3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all project data and results"""
    results_dir = Path("results/tcn_mlp_v2_evaluation")
    
    data = {
        'crop_performance': pd.read_csv(results_dir / "crop_performance.csv", encoding="utf-8"),
        'region_performance': pd.read_csv(results_dir / "region_performance.csv", encoding="utf-8"),
        'vulnerability': pd.read_csv(results_dir / "vulnerability_assessment.csv", encoding="utf-8"),
        'shap_importance': pd.read_csv(results_dir / "shap_feature_importance.csv", encoding="utf-8"),
        'strategies': pd.read_csv(results_dir / "adaptive_strategies_recommendations.csv", encoding="utf-8"),
        'detailed_vulnerability': pd.read_csv(results_dir / "detailed_vulnerability_matrix.csv", encoding="utf-8"),
    }
    
    # Load model metadata
    with open("models/TCN_MLP_L2_1e3_metadata.json", "r", encoding="utf-8") as f:
        data['metadata'] = json.load(f)
    
    return data

# Load evaluation summary
@st.cache_data
def load_summary():
    """Load evaluation summary text"""
    with open("results/tcn_mlp_v2_evaluation/EVALUATION_SUMMARY.txt", "r", encoding="utf-8") as f:
        return f.read()

# Initialize data
data = load_data()
summary_text = load_summary()

# Sidebar navigation
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Dashboard", "📊 Model Performance", "🌾 Crop Analysis", 
     "🗺️ Regional Vulnerability", "🔍 Feature Importance", "💡 Adaptive Strategies"]
)

# ==================== DASHBOARD PAGE ====================
if page == "🏠 Dashboard":
    st.title("🌾 Climate Change & Food Security in Nigeria")
    st.markdown("""
    **Evaluating the Impact of Climate Change and Soil on Food Security**
    
    This dashboard presents comprehensive analysis of crop yield predictions and climate vulnerability
    using advanced machine learning (TCN-MLP hybrid models) to inform adaptive strategies.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Performance (R²)",
            f"{0.8421:.4f}",
            "Excellent"
        )
    
    with col2:
        st.metric(
            "Mean Absolute Error",
            f"135.6 kg/ha",
            "±16.1% of mean yield"
        )
    
    with col3:
        st.metric(
            "Generalization Gap",
            "1.10%",
            "✓ Excellent"
        )
    
    # Key findings
    st.subheader("📈 Key Findings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🌾 Crop Performance**
        - Cassava: Mean = 819 kg/ha (CV = 111%)
        - Yams: Mean = 871 kg/ha (CV = 107%)
        - High yield variability indicates climate sensitivity
        """)
    
    with col2:
        st.warning("""
        **🌡️ Climate Vulnerability**
        - Cassava in South South: HIGH RISK (Score = 0.435)
        - Heat Stress in North East
        - Cold Stress in South West
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🗺️ Regional Performance**
        - South East: Highest yield (950 kg/ha)
        - South West: High yield (947 kg/ha)
        - North Central: Strong performance (927 kg/ha)
        """)
    
    with col2:
        st.info("""
        **💨 Top Climate Drivers**
        - Is_Rainy_Season (0.917)
        - Growing Degree Days (0.893)
        - Peak Growing Period (0.530)
        - Soil Moisture (0.428)
        """)
    
    # Crop performance overview
    st.subheader("🌾 Crop Yield Distribution")
    crop_perf = data['crop_performance']
    
    fig = px.bar(
        crop_perf,
        x='Crop',
        y='Mean Yield',
        error_y='Std Dev',
        title="Mean Yield by Crop",
        labels={'Mean Yield': 'Yield (kg/ha)'},
        color='Crop',
        color_discrete_map={'Cassava': '#D2B48C', 'Yams': '#8B7355'}
    )
    st.plotly_chart(fig, width='stretch')
    
    # Regional comparison
    st.subheader("🗺️ Regional Mean Yields")
    region_perf = data['region_performance']
    regional_yield = region_perf.groupby('Region')['Mean Yield'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=regional_yield.index,
        y=regional_yield.values,
        title="Average Yield by Region",
        labels={'x': 'Region', 'y': 'Mean Yield (kg/ha)'},
        color=regional_yield.values,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, width='stretch')


# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "📊 Model Performance":
    st.title("📊 TCN-MLP Model Performance")
    
    st.markdown("""
    The TCN-MLP hybrid architecture combines:
    - **Temporal Convolutional Network (TCN)**: Captures temporal dependencies in climate data
    - **Multi-Layer Perceptron (MLP)**: Processes derived features and interactions
    """)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test R² Score", "0.8421", "+0.0284 (5-fold CV)")
    with col2:
        st.metric("5-Fold CV R²", "0.8137 ± 0.0487", "Robust")
    with col3:
        st.metric("Test MAE", "135.6 kg/ha", "~16.1% error")
    with col4:
        st.metric("Generalization Gap", "1.10%", "Excellent")
    
    # Evaluation summary
    st.subheader("📋 Detailed Evaluation Summary")
    st.text(summary_text)
    
    # Model architecture info
    st.subheader("🏗️ Model Architecture")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **TCN Component:**
        - Kernel size: 3
        - Num levels: 3
        - Dropout rate: 0.2
        - Filters: 16, 32, 64
        """)
    
    with col2:
        st.info("""
        **MLP Component:**
        - Hidden layers: 128, 64
        - Activation: ReLU
        - Dropout rate: 0.2
        - Output: Single yield value
        """)
    
    # Training info
    st.subheader("🎓 Training Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Epochs", "100")
    with col2:
        st.metric("Batch Size", "32")
    with col3:
        st.metric("Optimizer", "Adam (lr=0.001)")


# ==================== CROP ANALYSIS PAGE ====================
elif page == "🌾 Crop Analysis":
    st.title("🌾 Crop Performance Analysis")
    
    crop_perf = data['crop_performance']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cassava")
        cassava_row = crop_perf[crop_perf['Crop'] == 'Cassava'].iloc[0]
        st.info(f"""
        **Yield Statistics:**
        - Count: {int(cassava_row['Count'])} observations
        - Mean: {cassava_row['Mean Yield']:.1f} kg/ha
        - Std Dev: {cassava_row['Std Dev']:.1f} kg/ha
        - CV: {cassava_row['CV (%)']:.1f}%
        - Range: {cassava_row['Min']:.1f} - {cassava_row['Max']:.1f} kg/ha
        
        **Interpretation:** High coefficient of variation indicates strong climate sensitivity
        """)
    
    with col2:
        st.subheader("Yams")
        yams_row = crop_perf[crop_perf['Crop'] == 'Yams'].iloc[0]
        st.info(f"""
        **Yield Statistics:**
        - Count: {int(yams_row['Count'])} observations
        - Mean: {yams_row['Mean Yield']:.1f} kg/ha
        - Std Dev: {yams_row['Std Dev']:.1f} kg/ha
        - CV: {yams_row['CV (%)']:.1f}%
        - Range: {yams_row['Min']:.1f} - {yams_row['Max']:.1f} kg/ha
        
        **Interpretation:** Yams show slightly better stability than cassava (lower CV)
        """)
    
    # Regional crop performance
    st.subheader("🗺️ Crop Performance by Region")
    
    vuln = data['vulnerability']
    crop_select = st.radio("Select Crop:", ["Cassava", "Yams"])
    crop_data = vuln[vuln['Crop'] == crop_select].sort_values('Mean_Yield', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            crop_data,
            x='Region',
            y='Mean_Yield',
            title=f"{crop_select} Mean Yield by Region",
            color='Mean_Yield',
            color_continuous_scale="RdYlGn",
            labels={'Mean_Yield': 'Yield (kg/ha)'}
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(
            crop_data,
            x='Region',
            y='Yield_CV',
            title=f"{crop_select} Yield Variability by Region",
            color='Yield_CV',
            color_continuous_scale="Reds",
            labels={'Yield_CV': 'Coefficient of Variation'}
        )
        st.plotly_chart(fig, width='stretch')
    
    # Detailed table
    st.subheader("📊 Detailed Regional Performance")
    st.dataframe(crop_data, width='stretch')


# ==================== REGIONAL VULNERABILITY PAGE ====================
elif page == "🗺️ Regional Vulnerability":
    st.title("🗺️ Regional Climate Vulnerability Analysis")
    
    vuln = data['vulnerability']
    
    # Vulnerability score heatmap
    st.subheader("🔥 Vulnerability Heatmap")
    pivot_data = vuln.pivot_table(
        index='Region',
        columns='Crop',
        values='Vulnerability_Score'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale="RdYlGn_r",
        text=np.round(pivot_data.values, 3),
        texttemplate="%{text:.3f}",
        textfont={"size": 12},
        colorbar=dict(title="Vulnerability<br>Score")
    ))
    fig.update_layout(
        title="Crop-Region Vulnerability Matrix",
        xaxis_title="Crop",
        yaxis_title="Region",
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    # Climate sensitivity analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Climate Sensitivity")
        fig = px.box(
            vuln,
            x='Crop',
            y='Climate_Sensitivity',
            color='Crop',
            title="Climate Sensitivity Distribution by Crop",
            labels={'Climate_Sensitivity': 'Sensitivity Score'}
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🌐 Exposure Index")
        fig = px.box(
            vuln,
            x='Crop',
            y='Exposure',
            color='Crop',
            title="Climate Exposure Distribution by Crop",
            labels={'Exposure': 'Exposure Score'}
        )
        st.plotly_chart(fig, width='stretch')
    
    # Highest risk areas
    st.subheader("⚠️ Highest Risk Areas")
    top_risks = vuln.nlargest(10, 'Vulnerability_Score')[
        ['Crop', 'Region', 'Vulnerability_Score', 'Climate_Sensitivity', 'Exposure']
    ]
    
    for idx, row in top_risks.iterrows():
        risk_level = "🔴 HIGH" if row['Vulnerability_Score'] > 0.5 else "🟡 MEDIUM"
        st.markdown(f"""
        {risk_level} **{row['Crop']} in {row['Region']}**
        - Vulnerability: {row['Vulnerability_Score']:.4f}
        - Climate Sensitivity: {row['Climate_Sensitivity']:.4f}
        - Exposure: {row['Exposure']:.4f}
        """)
    
    # Regional summary
    st.subheader("📊 Regional Summary Statistics")
    regional_stats = vuln.groupby('Region').agg({
        'Vulnerability_Score': ['mean', 'min', 'max'],
        'Climate_Sensitivity': 'mean',
        'Exposure': 'mean',
        'Mean_Yield': 'mean'
    }).round(4)
    st.dataframe(regional_stats, width='stretch')


# ==================== FEATURE IMPORTANCE PAGE ====================
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance & Model Interpretability")
    
    st.markdown("""
    These features were identified using SHAP (SHapley Additive exPlanations) values,
    providing model-agnostic explanations of feature contributions to yield predictions.
    """)
    
    shap_imp = data['shap_importance'].sort_values('Importance', ascending=True)
    
    # Feature importance bar chart
    fig = px.bar(
        shap_imp,
        y='Feature',
        x='Importance',
        orientation='h',
        title="Top 15 Features by SHAP Importance",
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale="Blues"
    )
    fig.update_xaxes(title_text="SHAP Importance Score")
    st.plotly_chart(fig, width='stretch')
    
    # Feature categories
    st.subheader("📋 Feature Categories & Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🌧️ Rainfall Features**
        - Is_Rainy_Season: 0.917 (Top)
        - SPI-3: ~0.3
        - Rainfall Interaction: 0.217
        
        **Impact:** Critical for yield prediction
        """)
    
    with col2:
        st.info("""
        **🌡️ Temperature Features**
        - GDD (Growing Degree Days): 0.893
        - Temperature: 0.223
        - Heat Stress: 0.201
        - Cold Stress: 0.205
        
        **Impact:** Strong temporal patterns
        """)
    
    with col3:
        st.info("""
        **🌱 Soil & Timing**
        - Soil Moisture: 0.428
        - Peak Growing Period: 0.530
        - Interactions: 0.640
        
        **Impact:** Non-linear relationships
        """)
    
    # Feature insights
    st.subheader("💡 Key Insights")
    insights = """
    1. **Rainfall Dominance**: Is_Rainy_Season is the single most important feature (0.917),
       suggesting seasonal precipitation patterns are critical for yield.
    
    2. **Thermal Accumulation**: Growing Degree Days (0.893) captures cumulative temperature effects,
       crucial for phenological development.
    
    3. **Soil-Climate Interactions**: Combined soil and climate features show that interactions matter
       more than individual features (0.64 vs ~0.42).
    
    4. **Stress Indicators**: Both heat stress (0.201) and cold stress (0.205) contribute significantly,
       indicating temperature extremes affect yields differently by crop/region.
    
    5. **Phenological Timing**: Peak growing period (0.530) shows timing matters - same conditions
       have different impacts at different growth stages.
    """
    st.markdown(insights)
    
    # Top features table
    st.subheader("📊 Complete Feature Rankings")
    st.dataframe(shap_imp.sort_values('Importance', ascending=False), width='stretch')


# ==================== ADAPTIVE STRATEGIES PAGE ====================
elif page == "💡 Adaptive Strategies":
    st.title("💡 Adaptive Strategies & Recommendations")
    
    strategies = data['strategies']
    
    # Risk level selector
    st.subheader("🎯 Filter by Risk Level")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        show_high = st.checkbox("High Risk (>0.50)", value=True)
    with col2:
        show_medium = st.checkbox("Medium Risk (0.45-0.50)", value=True)
    with col3:
        show_low = st.checkbox("Low Risk (<0.45)", value=True)
    
    # Filter strategies
    filtered_strategies = strategies.copy()
    if not (show_high and show_medium and show_low):
        masks = []
        if show_high:
            masks.append(filtered_strategies['Vulnerability_Score'] > 0.50)
        if show_medium:
            masks.append((filtered_strategies['Vulnerability_Score'] >= 0.45) & 
                        (filtered_strategies['Vulnerability_Score'] <= 0.50))
        if show_low:
            masks.append(filtered_strategies['Vulnerability_Score'] < 0.45)
        filtered_strategies = filtered_strategies[pd.concat(masks, axis=1).any(axis=1)]
    
    # Display strategies by region
    for region in filtered_strategies['Region'].unique():
        region_strats = filtered_strategies[filtered_strategies['Region'] == region]
        st.subheader(f"🗺️ {region}")
        
        for idx, row in region_strats.iterrows():
            risk_score = row['Vulnerability_Score']
            if risk_score > 0.50:
                risk_emoji = "🔴 HIGH RISK"
                risk_color = "danger"
            elif risk_score > 0.45:
                risk_emoji = "🟡 MEDIUM RISK"
                risk_color = "warning"
            else:
                risk_emoji = "🟢 LOW RISK"
                risk_color = "success"
            
            with st.expander(f"{risk_emoji} {row['Crop']} (Score: {risk_score:.4f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Vulnerability Details:**
                    - Score: {risk_score:.4f}
                    - Dominant Stressor: {row['Dominant_Stressor']}
                    - Early Warning: {row['Early_Warning_System']}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Recommended Actions:**
                    1. {row['Primary_Action_1']}
                    2. {row['Primary_Action_2']}
                    3. {"N/A" if pd.isna(row['Primary_Action_3']) or row['Primary_Action_3'] == 'N/A' else row['Primary_Action_3']}
                    """)
    
    # General recommendations
    st.subheader("🌍 Universal Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Immediate Actions:**
        1. Deploy TCN-MLP early warning systems for seasonal forecasts
        2. Implement crop variety diversification programs
        3. Establish soil moisture monitoring networks
        4. Train farmers on stress detection and management
        """)
    
    with col2:
        st.info("""
        **📋 Long-term Strategies:**
        1. Develop climate-smart agriculture programs
        2. Invest in irrigation infrastructure
        3. Create regional seed banks with heat/cold tolerant varieties
        4. Establish climate information services (CIS)
        5. Promote agro-forestry and soil conservation
        """)
    
    # Summary statistics
    st.subheader("📊 Strategy Coverage Summary")
    col1, col2, col3 = st.columns(3)
    
    high_risk_count = len(strategies[strategies['Vulnerability_Score'] > 0.50])
    early_warning_count = len(strategies[strategies['Early_Warning_System'].str.contains('YES|RECOMMENDED', na=False)])
    
    with col1:
        st.metric("High Risk Areas", high_risk_count)
    with col2:
        st.metric("Early Warning Deployment", early_warning_count)
    with col3:
        st.metric("Total Crop-Region Units", len(strategies))


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Project Information
**Research Focus:** Climate Change & Food Security in Nigeria

**Model:** TCN-MLP Hybrid Architecture
- R² Score: 0.8421
- MAE: 135.6 kg/ha

**Data:** 1,728 observations per crop
- Crops: Cassava, Yams
- Regions: 6 regions across Nigeria
- Features: Climate, soil, and phenological

**Contact & Attribution**
Climate Change & Food Security Research Project
""")
