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
    
    st.subheader("🛠️ How to Use This Application")
    
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
