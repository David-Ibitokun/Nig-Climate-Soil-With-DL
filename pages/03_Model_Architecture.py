import streamlit as st
from data_loader import apply_global_style, load_data


def render():
    apply_global_style()
    
    st.title("🏗️ Model Architecture")
    
    st.markdown("""
Learn about the TCN-MLP Ensemble architecture that powers the crop yield predictions.
    """)
    
    st.markdown("---")
    
    # Architecture Overview
    st.subheader("🔍 Architecture Overview")
    
    st.markdown("""
The model combines two complementary neural network components:

### **1. Temporal Convolutional Network (TCN)**
- **Purpose**: Captures temporal dependencies in climate sequences
- **Functionality**:
  - Processes sequential climate data across growing seasons
  - Identifies which time periods are most critical for yield
  - Uses dilated convolutions to capture both short and long-term patterns
  - Learns hierarchical representations of seasonal patterns

### **2. Multi-Layer Perceptron (MLP)**
- **Purpose**: Combines TCN features with spatial context
- **Functionality**:
  - Processes extracted temporal patterns from TCN
  - Integrates crop-specific and regional information
  - Learns complex nonlinear interactions between features
  - Produces final yield predictions

### **Ensemble Approach**
- **Multiple Models**: 5-fold cross-validation creates diverse models
- **Ensemble Aggregation**: Predictions averaged across folds for robustness
- **Confidence Intervals**: Ensemble variance provides prediction uncertainty
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Data Flow")
    
    st.markdown("""
```
Climate Sequences (Temperature, Rainfall, Humidity, etc.)
            ↓
    Temporal Convolutional Network (TCN)
            ↓
    Hierarchical Temporal Features
            ↓
    Multi-Layer Perceptron (MLP)
            ↓
    Crop Yield Prediction + Confidence Interval
```
    """)
    
    st.markdown("---")
    
    st.subheader("🎯 Key Design Choices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
#### **Why TCN?**
- Preserves temporal order of climate data
- Efficient memory usage with dilated convolutions
- Can capture both short-term weather shocks and long-term seasonal patterns
- Interpretable time windows for feature importance
        """)
    
    with col2:
        st.markdown("""
#### **Why Ensemble?**
- Reduces overfitting through diverse models
- Provides uncertainty estimates
- More stable predictions across different data splits
- Better generalization to new regions/years
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Model Performance")
    
    data = load_data()
    meta = data.get("metadata") or {}
    perf = meta.get("performance", {}) if isinstance(meta, dict) else {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Test R² Score",
            value=f"{perf.get('final_test_r2', 0):.4f}",
            help="Proportion of variance explained"
        )
    with col2:
        st.metric(
            label="Test MAE",
            value=f"{perf.get('final_test_mae', 0):.1f} kg/ha",
            help="Mean absolute error on test set"
        )
    with col3:
        st.metric(
            label="CV Validation MAE",
            value=f"{perf.get('cv_val_mae_mean', 0):.1f} kg/ha",
            help="Cross-validation performance"
        )
    with col4:
        st.metric(
            label="Crops Trained",
            value="4",
            help="Maize, Rice, Cassava, Yam"
        )
    
    st.markdown("---")
    
    st.subheader("💡 Feature Engineering")
    
    st.markdown("""
The model uses several derived features:

1. **Temporal Aggregates**: Rolling means and stds of climate variables
2. **Interaction Features**: Temperature × Rainfall interactions
3. **Seasonal Indices**: Month/season encoding for seasonal patterns
4. **Lagged Features**: Previous season climate effects
5. **Regional Context**: Regional dummy variables for localized effects
    """)


if __name__ == "__main__":
    render()
