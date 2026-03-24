# 🌾 Climate Change & Food Security in Nigeria - Streamlit App

## Overview

Interactive dashboard analyzing the impact of climate change and soil conditions on food security in Nigeria using advanced TCN-MLP machine learning models.

## Features

📊 **6 Interactive Pages:**

1. **Dashboard** - High-level overview with key metrics and findings
   - Model performance summary
   - Crop yield distribution
   - Regional comparison charts

2. **Model Performance** - TCN-MLP architecture and evaluation metrics
   - R² scores and cross-validation results
   - Model architecture details
   - Training configuration

3. **Crop Analysis** - Detailed crop yield statistics
   - Cassava and Yams performance
   - Yield distributions by region
   - Coefficient of variation analysis

4. **Regional Vulnerability** - Climate vulnerability assessment
   - Crop-region vulnerability heatmap
   - Climate sensitivity and exposure
   - Highest risk areas identification

5. **Feature Importance** - Model interpretability using SHAP
   - Top predictive features
   - Feature categories and impact
   - Rainfall, temperature, and soil factors

6. **Adaptive Strategies** - Region-specific recommendations
   - Risk-based filterable strategies
   - Dominant stressors and actions
   - Early warning system deployment

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Navigate to project directory:**
   ```bash
   cd c:\Users\ibito\Downloads\project_test
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Command Line Options

```bash
# Disable browser auto-opening
streamlit run app.py --logger.level=error

# Run on specific port
streamlit run app.py --server.port 8502

# Run in headless mode
streamlit run app.py --server.headless true
```

## App Navigation

- **Sidebar Menu**: Use the radio buttons to navigate between pages
- **Interactive Charts**: Hover over charts for details, use zoom/pan tools
- **Expandable Sections**: Click on region/crop headers to expand strategy details
- **Filterable Results**: Use checkboxes to filter strategies by risk level

## Data Sources

All analysis data is loaded from:
- `results/tcn_mlp_v2_evaluation/` - Evaluation results and visualizations
- `models/TCN_MLP_L2_1e3_metadata.json` - Model metadata
- `project_data/processed_data/` - Original datasets

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Test R² | 0.8421 | 84.2% variance explained |
| 5-Fold CV R² | 0.8137 ± 0.0487 | Robust across folds |
| Test MAE | 135.6 kg/ha | ~16.1% mean error |
| Generalization Gap | 1.10% | Excellent generalization |

## Crop Performance Summary

### Cassava
- Mean Yield: 819 kg/ha
- Coefficient of Variation: 111%
- Range: 0-2,687 kg/ha
- High climate sensitivity

### Yams
- Mean Yield: 871 kg/ha
- Coefficient of Variation: 107%
- Range: 0-3,736 kg/ha
- More stable than cassava

## Regional Insights

**Highest Performing Regions:**
1. South East: 950 kg/ha average
2. South West: 947 kg/ha average
3. North Central: 927 kg/ha average

**Highest Risk Areas:**
- Cassava in South South (Vulnerability: 0.435)
- Heat stress in North East regions
- Cold stress in South West regions

## Top Climate Drivers

Features ranked by SHAP importance:
1. Is_Rainy_Season (0.917) - Critical
2. Growing Degree Days (0.893) - Essential
3. Peak Growing Period (0.530) - Important
4. Soil Moisture (0.428) - Important
5. Temperature interactions (0.42+) - Key

## Adaptive Strategies

All recommendations include:
- Risk assessment (vulnerability score)
- Dominant climate stressor identification
- Primary recommended actions
- Early warning system deployment status

**Universal Recommendations:**
- Diversify crop varieties by region
- Deploy TCN-MLP early warning systems
- Implement soil moisture monitoring
- Establish seasonal climate forecasts

## Troubleshooting

### App Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify Streamlit installation
pip list | grep streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

### Data Loading Errors
- Ensure you're in the correct directory: `c:\Users\ibito\Downloads\project_test`
- Check that all result files exist in `results/tcn_mlp_v2_evaluation/`
- Verify CSV files are properly formatted

### Performance Issues
- Close other applications
- Reduce browser tab count
- Use `--logger.level=error` for faster startup
- Clear browser cache (Ctrl+Shift+Delete)

## Browser Compatibility

- ✅ Chrome/Chromium (Recommended)
- ✅ Firefox
- ✅ Edge
- ✅ Safari
- ⚠️ IE (Not supported)

## Caching

The app uses Streamlit's caching decorator (`@st.cache_data`) for:
- Data loading from CSVs
- Summary text loading
- Metadata JSON loading

Changes to underlying data files will require clearing cache or restarting the app.

## Customization

To modify the app:

1. **Edit color schemes** - Look for `color_discrete_map` in chart definitions
2. **Adjust metrics displayed** - Modify the Dashboard page section
3. **Add new visualizations** - Create new sections in the appropriate pages
4. **Change layout** - Adjust `st.columns()` settings for responsive design

## Performance Notes

- Initial load: ~2-3 seconds (data caching)
- Subsequent loads: <1 second
- Interactive charts render on hover/interaction
- Heatmaps optimized for 6 regions × 2 crops

## Contact & Support

For issues or questions about:
- **Data Analysis** - See research notebooks
- **Model Details** - Check `TCN_MLP_v2_Comprehensive_Evaluation.ipynb`
- **Raw Results** - Review files in `results/tcn_mlp_v2_evaluation/`

---

**Last Updated:** March 2026
**Model Version:** TCN-MLP v2 (L2 Regularization: 1e-3)
**Data Version:** 4.1 Clean
