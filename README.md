# Climate Change & Food Security in Nigeria - TCN-MLP Analysis

A comprehensive machine learning analysis project investigating the impact of climate change on food security in Nigeria using Temporal Convolutional Networks (TCN) and Multi-Layer Perceptron (MLP) hybrid models.

## Project Overview

This project employs advanced deep learning techniques to:
- Analyze climate-food security relationships in Nigeria
- Predict crop yield patterns
- Assess regional vulnerability to climate variability
- Generate adaptive strategies and recommendations

## Dataset & Data Sources

- **Project Data**: Located in `project_data/`
  - Raw sources in `data_sources.csv`
  - Processed datasets in `processed_data/`
  - Includes soil data, crop yields, and climate variables

## Models

### TCN-MLP Hybrid Architecture

The project uses a sophisticated hybrid model combining:
- **Temporal Convolutional Network (TCN)**: Captures temporal dependencies and patterns
- **Multi-Layer Perceptron (MLP)**: Processes derived features

**Trained Models**:
- `models/TCN_MLP_L2_1e3_Best.keras` - Best performing model
- `models/v2_saved/` - Version 2 models

## Notebooks

1. **TCN_MLP_v2.ipynb** - Initial model implementation and training
2. **TCN_MLP_v2_Comprehensive_Evaluation.ipynb** - Detailed model evaluation and performance metrics
3. **TCN_MLP_Best_Configuration.ipynb** - Hyperparameter optimization and configuration analysis
4. **climate_change_food_security_nigeria_v4.1_clean.ipynb** - Data cleaning and exploratory analysis

## Results & Outputs

All results are stored in `results/tcn_mlp_v2_evaluation/`:

- **EVALUATION_SUMMARY.txt** - High-level performance metrics
- **crop_performance.csv** - Crop-specific predictions and evaluations
- **region_performance.csv** - Regional analysis and metrics
- **vulnerability_assessment.csv** - Climate vulnerability by region
- **detailed_vulnerability_matrix.csv** - Comprehensive vulnerability analysis
- **shap_feature_importance.csv** - SHAP-based feature importance rankings
- **adaptive_strategies_recommendations.csv** - Recommended adaptation strategies

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- TensorFlow/Keras
- NumPy & Pandas
- Scikit-learn
- SHAP (for model interpretability)

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project_test
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run notebooks:
   ```bash
   jupyter notebook
   ```

## Project Structure

```
project_test/
├── README.md
├── requirements.txt
├── .gitignore
├── TCN_MLP_v2.ipynb
├── TCN_MLP_v2_Comprehensive_Evaluation.ipynb
├── TCN_MLP_Best_Configuration.ipynb
├── climate_change_food_security_nigeria_v4.1_clean.ipynb
├── models/
│   ├── TCN_MLP_L2_1e3_Best.keras
│   ├── TCN_MLP_L2_1e3_metadata.json
│   └── v2_saved/
├── project_data/
│   ├── data_sources.csv
│   └── processed_data/
│       ├── annual_yield_hybrid_enhanced.csv
│       ├── annual_yield_tcn_mlp_data.csv
│       ├── master_data_*.csv
│       ├── tcn_mlp_dataset.csv
│       ├── tcn_mlp_soil_data*.csv
│       └── preprocessing_metadata*.json
└── results/
    └── tcn_mlp_v2_evaluation/
        ├── EVALUATION_SUMMARY.txt
        ├── crop_performance.csv
        ├── region_performance.csv
        ├── vulnerability_assessment.csv
        ├── detailed_vulnerability_matrix.csv
        ├── shap_feature_importance.csv
        └── adaptive_strategies_recommendations.csv
```

## Key Features

- **Temporal Analysis**: TCN captures sequential climate patterns
- **Hybrid Architecture**: Combines temporal and feature-based learning
- **Interpretability**: SHAP analysis for feature importance
- **Regional Insights**: Separate evaluation by Nigerian regions
- **Vulnerability Assessment**: Climate resilience and adaptation recommendations

## Usage Examples

### Load and Use the Trained Model

```python
from tensorflow import keras
import numpy as np

# Load the best model
model = keras.models.load_model('models/TCN_MLP_L2_1e3_Best.keras')

# Make predictions on new data
predictions = model.predict(your_data)
```

### Analyze Results

```python
import pandas as pd

# Load evaluation results
eval_summary = pd.read_csv('results/tcn_mlp_v2_evaluation/crop_performance.csv')
vulnerability = pd.read_csv('results/tcn_mlp_v2_evaluation/vulnerability_assessment.csv')
```

## Results Summary

Key findings are documented in:
- `results/THESIS_RESULTS.txt` - Comprehensive thesis results
- `results/tcn_mlp_v2_evaluation/EVALUATION_SUMMARY.txt` - Model evaluation summary

## Contributing

For modifications and improvements:
1. Create a new branch
2. Make your changes in a new notebook or modify existing ones
3. Document results in the `results/` directory
4. Commit with clear messages

## Author

David Ibitokun 

---

**Last Updated**: March 2026
