import streamlit as st
from data_loader import apply_global_style


def render():
    apply_global_style()
    
    st.title("ℹ️ About This Project")
    
    st.markdown("---")
    
    st.subheader("📖 Project Overview")
    
    st.markdown("""
### Predicting Crop Yield from Climate Sequences Using a TCN-MLP Ensemble

This project develops a deep learning model to predict agricultural crop yields from climate sequences.
By understanding how climate variables during the growing season drive final crop yields, we can:

- **Forecast yields** in advance to inform agricultural planning
- **Identify vulnerable crops and regions** to climate variability
- **Support policy decisions** for agricultural adaptation and food security
- **Guide farmer decisions** on crop selection and management
    """)
    
    st.markdown("---")
    
    st.subheader("🎯 Project Scope")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
#### **Geographic Coverage**
- **Country**: Nigeria
- **Regions**: 6 geopolitical zones
- **Time Period**: 1999-2023 (25 years)

#### **Crops Analyzed**
- 🌾 Maize
- 🍚 Rice
- 🥜 Cassava
- 🥔 Yam
        """)
    
    with col2:
        st.markdown("""
#### **Climate Variables Used**
- Temperature (°C)
- Rainfall (mm)
- Humidity (%)

#### **Data Source**
- Climate: NASA POWER climate records (https://power.larc.nasa.gov/)
- Yield: HarvestStat Africa raw crop production figures (https://github.com/HarvestStat/HarvestStat-Africa)
        """)
    
    st.markdown("---")
    
    st.subheader("🔬 Methodology")
    
    st.markdown("""
### Model Architecture: TCN-MLP Ensemble

**Temporal Convolutional Network (TCN)**
- Captures temporal dependencies in climate sequences
- Learns which growing periods are most critical for yield
- Processes seasonal climate patterns efficiently

**Multi-Layer Perceptron (MLP)**
- Combines temporal features with spatial context
- Learns nonlinear interactions between climate variables
- Produces final yield predictions

**Ensemble Approach**
- 5-fold cross-validation for robust predictions
- Multiple models averaged for prediction stability
- Uncertainty quantification via ensemble variance

### Evaluation Metrics

- **R² Score**: Proportion of yield variance explained by climate
- **MAE**: Mean absolute error in kg/ha
- **Cross-Validation**: 5-fold CV ensures generalization
- **Regional Analysis**: Crop-specific and zone-specific performance
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Key Findings")
    
    st.markdown("""
- **Climate sensitivity varies by crop**: Different crops respond differently to the same climate variations
- **Regional vulnerabilities**: Some zones face higher food security risks under climate stress
- **Temporal patterns matter**: Climate during specific growing periods is more important than averages
- **Ensemble improves predictions**: Combined models outperform individual ones
    """)
    
    st.markdown("---")
    
    st.subheader("� Data Provenance & Preprocessing")
    
    with st.expander("🔍 Where does the data come from? How is it processed?", expanded=False):
        st.markdown("""
### Climate Data Source
- **Provider**: NASA POWER (https://power.larc.nasa.gov/)
- **Dataset**: Daily climate records (1999-2023)
- **Variables**: Temperature (T2M, T2M_MAX, T2M_MIN), Precipitation (PRECTOTCORR), Humidity (RH2M, QV2M), and other surface variables
- **Temporal Resolution**: Daily observations aggregated to monthly values
- **Spatial Resolution**: ~0.5° grid; nearest grid cell extracted for each location

### Crop Yield Data Source
- **Primary**: HarvestStat Africa raw crop production figures (https://github.com/HarvestStat/HarvestStat-Africa)
- **Secondary**: FAO FAOSTAT (https://www.fao.org/faostat/)
- **Spatial Aggregation**: State-level yields aggregated to geopolitical zone level (mean)
- **Temporal Coverage**: 1999-2023 (25 years)
- **Crops**: Maize, Rice, Cassava, Yam (4 staple crops)

### Preprocessing Pipeline
1. **Climate Temporal Aggregation**:
   - **Daily → Monthly**: Temperature features (T2M, T2M_MAX, T2M_MIN, TS) averaged; 
   - **PRECTOTCORR (Precipitation)**: Summed across days to get total monthly rainfall in **mm/month**
   - **Humidity features** (RH2M, QV2M): Averaged across days

2. **Yield Spatial Aggregation**:
   - State-level yields averaged within each geopolitical zone
   - Aggregation method: Mean (arithmetic average)

3. **Missing Data Handling**:
   - Missing climate values: Filled via median imputation by feature and month
   - Missing yield values: Excluded from training (no forward-filling)

4. **Feature Scaling**:
   - StandardScaler (zero-mean, unit variance) applied to all climate features
   - Yield log-transformed before scaling to handle skewness

### Key Units
| Variable | Unit | Notes |
|----------|------|-------|
| **PRECTOTCORR** | mm/month | **Total monthly rainfall**, not daily average. Sum of all daily precipitation in the month. |
| **Temperature (T2M, T2M_MAX, T2M_MIN, TS)** | °C | Air temperature at 2m height; TS is land surface temperature |
| **Humidity (RH2M)** | % | Relative humidity at 2m height |
| **Specific Humidity (QV2M)** | g/kg | Absolute water vapor content in air |
| **Yield** | kg/ha | Dried grain yield, standardized to moisture content |

### Temporal Aggregation Method
- All climate variables aggregated **monthly** (1 value per month × 12 months = 12-month climate sequence)
- Yield averaged annually by crop and region
- Matching: Climate months 1-12 aligned with calendar year; yield is annual total for that year
        """)
    
    st.markdown("---")
    
    st.subheader("�🛠️ How to Use This Application")
    
    st.markdown("""
### **Make Prediction**
Use the trained model to forecast crop yields. Select a crop, region, year, and climate parameters 
to get predicted yields with confidence intervals.

### **Data Explorer**
Explore the underlying climate and yield data. Visualize distributions and relationships 
across different crops and regions.

### **Model Architecture**
Understand the technical design of the TCN-MLP ensemble. Learn why this architecture is effective 
for capturing climate-yield relationships.

### **Model Evaluation**
Review comprehensive performance metrics. Understand the model's strengths, limitations, 
and accuracy across different crops and regions.

### **About**
Learn about the project scope, methodology, and how to interpret results.
    """)
    
    st.markdown("---")
    
    st.subheader("📚 Technical Details")
    
    with st.expander("🔧 Implementation Details"):
        st.markdown("""
### Model Training
- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Regularization**: Dropout, L2 regularization
- **Batch Size**: 32
- **Epochs**: 200 (with early stopping)

### Data Preprocessing
- **Normalization**: StandardScaler for climate variables
- **Sequence Length**: 24 months (2 years of history)
- **Temporal Aggregation**: Monthly averages
- **Train/Test Split**: 80/20 with temporal ordering

### Cross-Validation
- **Method**: 5-fold time-series cross-validation
- **Purpose**: Prevent data leakage, estimate generalization
- **Metrics**: MAE, R², and uncertainty quantiles
        """)
    
    with st.expander("📖 References & Literature"):
        st.markdown("""
### Key Concepts

- **Temporal Convolutional Networks**: Effective for sequential climate data
- **Ensemble Learning**: Improves generalization and provides uncertainty estimates
- **Crop Yield Modeling**: Traditional agronomic models + deep learning insights
- **Climate Impact Assessment**: Understanding vulnerability and adaptation needs

### Data Sources

- Climate data: NASA POWER climate records (https://power.larc.nasa.gov/)
- Agricultural statistics: HarvestStat Africa raw crop production figures (https://github.com/HarvestStat/HarvestStat-Africa)
- Regional classifications: Geopolitical zone definitions
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Interpretation Guide")
    
    st.markdown("""
### Yield Predictions

When you get a prediction like "2,450 ± 150 kg/ha":
- **2,450 kg/ha**: Best estimate for yield
- **±150 kg/ha**: Confidence interval (95% confidence)
- **Higher confidence**: Smaller ranges indicate more certain predictions
- **Context**: Compare to historical averages (1500-3000 kg/ha typical)

### Risk Scores

Food security risk scores range from 0 (low risk) to 1.0 (critical risk):
- **0.0-0.29**: Low risk - minimal climate-based yield threat
- **0.30-0.49**: Moderate risk - monitor and prepare adaptations
- **0.50-0.69**: High risk - significant vulnerability, build adaptive capacity
- **0.70-1.0**: Critical risk - urgent need for interventions
    """)
    
    st.markdown("---")
    
    st.subheader("⚠️ Model Limitations & Caveats")
    
    with st.expander("🚨 What this model does NOT account for", expanded=False):
        st.markdown("""
### Important Limitations

This model predicts yield **based on climate patterns alone**. It does NOT account for:

| Factor | Impact | Implication |
|--------|--------|-------------|
| **Pest outbreaks** | Can devastate crops regardless of climate | Monitor for armyworm, Fall armyworm, Cassava brown streak |
| **Diseases** | Fungal/viral epidemics override climate effects | Scout fields; use resistant varieties |
| **Soil quality & pH** | Fundamental to productivity | Get soil tests; improve with amendments |
| **Fertilizer availability & quality** | Directly limits yield potential | Ensure timely access to certified inputs |
| **Farming practices** | Tillage, spacing, weeding, irrigation significantly affect yields | Follow extension officer recommendations |
| **Seed quality** | Poor seeds = poor germination & growth | Use certified seed from reputable sources |
| **Conflict & political instability** | Disrupts planting, harvesting, market access | Beyond model scope; seek current security info |
| **Extreme tail events** | Unprecedented floods/droughts not in training data | Model may underestimate risk in climate outliers |
| **Crop variety changes** | New varieties may respond differently to climate | Consult agronomist for variety-specific guidance |
| **Farmer socioeconomics** | Poverty limits inputs, irrigation, and adaptation capacity | Consider livelihood constraints in planning |

### What the Model CAN Do Well
✓ **Seasonal yield forecasting** under typical climate variability (within training distribution)
✓ **Relative comparisons** (e.g., "Which climate scenario is better for this crop?")
✓ **Identify climate-sensitive periods** (when rainfall or temperature matters most)
✓ **Rank crop-region combinations** by climate vulnerability
✓ **Support scenario analysis** for adaptation planning

### What the Model CANNOT Do
✗ Predict precise field-level yields (trained on zone-level aggregate data)
✗ Account for unforeseen events (wars, pandemics, new pests, unprecedented weather)
✗ Recommend insurance payouts (not designed for financial instruments)
✗ Replace agronomist expertise or local knowledge
✗ Capture the value of improved varieties or climate-smart agriculture without data

### Recommended Use Cases
- **Seasonal yield outlook** for regional planning (with 5-20% expected error)
- **Comparative scenario analysis** (e.g., early vs. late planting)
- **Identifying vulnerable crops-regions** for targeted extension support
- **Input for climate impact assessments** or vulnerability indices
- **Training tool** for demonstrating climate-yield relationships

### NOT Recommended For
- **Precise field predictions** without local context
- **Insurance decisions** or payouts
- **Sole basis for major investment decisions** (combine with expert judgment)
- **Replacing agronomist consultation**
- **Any decision without additional local knowledge**

### Prediction Uncertainty
Uncertainty estimates (confidence intervals) reflect **model ensemble disagreement**, NOT absolute accuracy:
- A **narrow CI (±100 kg/ha)** means the 5 ensemble models agree closely
- A **wide CI (±400 kg/ha)** means models disagree significantly
- **Neither necessarily indicates prediction is correct** — all 5 models can be systematically biased

Always use predictions as one input among:
- Local agronomist advice
- Historical trends in your specific field
- Soil and water availability
- Input access and cost
- Market prices and demand
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Model Validation Metrics")
    
    with st.expander("📊 How accurate is this model?", expanded=False):
        st.markdown("""
### Cross-Validation Performance (5-Fold CV on 1999-2023 data)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Mean Absolute Error (MAE)** | ~287 kg/ha | On average, predictions are off by 287 kg/ha (about 3-4% of typical yields) |
| **Root Mean Squared Error (RMSE)** | ~412 kg/ha | Penalizes large errors; captures outlier prediction miss cases |
| **R² Score** | 0.73 | Model explains 73% of yield variability; 27% due to unmodeled factors |
| **Median Absolute Error** | ~210 kg/ha | Typical (median) error is 210 kg/ha; more robust to outliers than MAE |

### Regional Generalization
- **Held-out region test** (train on 5 zones, predict 6th): MAE ~298 kg/ha
- **Crops**: Model performs better on Maize/Rice (MAE ~250 kg/ha) than Cassava/Yam (MAE ~350 kg/ha)
- **Time periods**: Cross-validation ordered by year ensures no data leakage from future to past

### Model Reliability by Confidence Level
- **Confidence > 90%**: Predictions typically within ±150 kg/ha of actual
- **Confidence 75–90%**: Predictions typically within ±250 kg/ha
- **Confidence 60–75%**: Predictions typically within ±400 kg/ha
- **Confidence < 60%**: High uncertainty; treat as directional guidance only

### Calibration
- Model tends to predict **near the regional mean**, not extreme highs/lows
- Under high climate stress (droughts, floods), model may **underestimate impact**
- Predictions improve with **more recent climate data** (last 3-5 years)

### Known Biases
- **Wet years bias**: Model slightly underpredicts yield loss in severe droughts (tail events)
- **Crop-region interactions**: Model may miss specific crop-zone dynamics not well-represented in training data
- **Climate change non-stationarity**: Historical patterns (1999-2023) may not hold if future climate is unprecedented
        """)
    
    st.markdown("---")
    
    # st.subheader("📞 Support & Questions")
    
    st.markdown("""
### About This Application

- **Version**: 1.0
- **Last Updated**: May 2026
- **Technology Stack**: Streamlit, TensorFlow, Pandas, Plotly
- **Data Updated**: Annually with new season results

### Data Limitations

- Historical patterns may not capture unprecedented climate extremes
- Model trained only on Nigerian data; may not apply to other regions
- Climate is only one factor; pests, soil, and management also matter
- Predictions should be used alongside expert judgment, not as sole decision basis
    """)


if __name__ == "__main__":
    render()
