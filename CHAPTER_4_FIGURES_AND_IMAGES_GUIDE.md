# Chapter 4: Screenshots & Images Reference Guide

This document provides a comprehensive guide to all figures, tables, and visualizations referenced in Chapter 4 (Results and Discussion). Each entry includes:
- Figure/Table number and title
- Location in chapter (section number)
- Description of content and purpose
- Suggested dimensions and format
- How to generate or capture
- Reference to source files/notebooks

---

## FIGURES

### Figure 4.1: Observed vs. Predicted Scatter Plot with 1:1 Reference Line & Confidence Intervals

**Section:** 4.2.2 (Ensemble vs. Single-Model Performance)

**Purpose:** Visualize overall prediction accuracy; show systematic bias (if any) and prediction spread across the test set

**Description:**
- X-axis: Observed yield (kg/ha)
- Y-axis: Predicted yield (kg/ha)
- Points: Individual test samples, colored by region or crop
- Reference line: 1:1 diagonal (perfect prediction)
- CI bands: Shaded region showing ±1.96σ ensemble uncertainty around predicted values
- Marginal plots (optional): Histograms of observed and predicted distributions

**Suggested format:**
- Plotly interactive scatter or static Matplotlib/Seaborn
- Size: 8" × 6" (landscape, readable at book size)
- Color palette: Distinct colors per region (e.g., North-Central=blue, South-East=green)
- Annotations: R², MAE, RMSE displayed on plot

**How to generate:**
```python
# Pseudocode; see notebooks/tcn_mlp_eval.ipynb for full implementation
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
# Scatter plot
ax.scatter(y_test, y_pred, alpha=0.6, c=regions, cmap='tab10', s=50)
# 1:1 reference
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
# CI bands
ax.fill_between(y_test, y_pred - 1.96*y_std, y_pred + 1.96*y_std, alpha=0.2)
ax.set_xlabel('Observed Yield (kg/ha)')
ax.set_ylabel('Predicted Yield (kg/ha)')
ax.set_title(f'Observed vs. Predicted Yield\nR² = {r2:.3f}, MAE = {mae:.0f} kg/ha')
ax.legend()
plt.savefig('figure_4_1_obs_vs_pred.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` — evaluation section
**Output file:** `results/figures/figure_4_1_obs_vs_pred.png`

---

### Figure 4.2: Regional Performance Heatmap (MAE & RMSE by Crop and Region)

**Section:** 4.2.4 (Per-Region Performance)

**Purpose:** Visualize geographic heterogeneity in model performance; identify which crop-region combinations are predicted well vs. poorly

**Description:**
- Rows: 6 regions (North-Central, North-East, North-West, South-East, South-South, South-West)
- Columns: 4 crops (Maize, Rice, Cassava, Yam)
- Cell color intensity: MAE or RMSE value (darker = higher error)
- Cell annotation: MAE (kg/ha) and/or RMSE value
- Colorbar: Gradient (e.g., light=low error, dark=high error)

**Suggested format:**
- Seaborn heatmap or Plotly matrix
- Size: 10" × 6" (landscape)
- Color scheme: diverging (e.g., green=good, red=poor)
- Font: Clear cell labels for region and crop

**How to generate:**
```python
import seaborn as sns
import pandas as pd

# Create pivot table: rows=region, columns=crop, values=MAE
mae_pivot = results_df.pivot_table(
    values='MAE', 
    index='Region', 
    columns='Crop'
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    mae_pivot, 
    annot=True, 
    fmt='.0f', 
    cmap='RdYlGn_r', 
    cbar_kws={'label': 'MAE (kg/ha)'},
    ax=ax
)
ax.set_title('Model Performance by Region and Crop')
plt.savefig('figure_4_2_regional_heatmap.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` — per-region analysis
**Output file:** `results/figures/figure_4_2_regional_heatmap.png`

---

### Figure 4.3: Histogram of Uncertainty Percentage with Confidence Thresholds

**Section:** 4.3.2 (Uncertainty Distribution & Thresholds)

**Purpose:** Show distribution of model uncertainty across test predictions; visualize threshold bands (Low/Medium/High)

**Description:**
- X-axis: Uncertainty percentage (0–100%)
- Y-axis: Frequency (number of predictions in each bin)
- Histogram bars: Color-coded by uncertainty band:
  - Green (Low): < 10%
  - Yellow (Medium): 10–25%
  - Red (High): ≥ 25%
- Vertical lines: Threshold boundaries (10%, 25%)
- Statistics: Mean, median, std dev annotated on plot

**Suggested format:**
- Matplotlib histogram with color-coded bins
- Size: 8" × 5" (landscape)
- Bins: 20–30 bins for smooth distribution
- Overlay: KDE curve (optional, for smoothness)

**How to generate:**
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
uncertainty_pct = # from predictions

# Define bins and colors
bins = np.linspace(0, 100, 31)
colors = np.where(uncertainty_pct < 10, 'green', 
         np.where(uncertainty_pct < 25, 'yellow', 'red'))

ax.hist(uncertainty_pct, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(10, color='yellow', linestyle='--', linewidth=2, label='Low/Medium threshold')
ax.axvline(25, color='red', linestyle='--', linewidth=2, label='Medium/High threshold')
ax.set_xlabel('Uncertainty Percentage (%)')
ax.set_ylabel('Frequency (# predictions)')
ax.set_title(f'Distribution of Model Uncertainty\nMean = {uncertainty_pct.mean():.1f}%, Median = {np.median(uncertainty_pct):.1f}%')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.savefig('figure_4_3_uncertainty_histogram.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` — uncertainty analysis
**Output file:** `results/figures/figure_4_3_uncertainty_histogram.png`

---

### Figure 4.4: Calibration Curve (Observed Coverage vs. Nominal Confidence Interval Level)

**Section:** 4.3.4 (Calibration Analysis)

**Purpose:** Assess whether predicted confidence intervals are reliable; validate that 95% CIs contain ~95% of observations

**Description:**
- X-axis: Nominal CI level (0–100%, typically 50, 75, 90, 95%)
- Y-axis: Observed coverage (% of observations actually within predicted interval)
- Points: Empirical coverage at each CI level
- Reference line: 45° diagonal (perfect calibration; nominal = observed)
- Shaded region (optional): ±5% tolerance band around diagonal

**Suggested format:**
- Scatter plot + reference line
- Size: 8" × 8" (square, emphasizes diagonal)
- Markers: Large circles (e.g., 100 points)
- Color: Blue for points, black for diagonal

**How to generate:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Compute coverage at multiple nominal levels
nominal_levels = np.array([50, 75, 90, 95])
observed_coverage = []

for level in nominal_levels:
    z_score = scipy.stats.norm.ppf((1 + level/100) / 2)
    ci_lower = y_pred - z_score * y_std
    ci_upper = y_pred + z_score * y_std
    coverage = np.mean((y_test >= ci_lower) & (y_test <= ci_upper)) * 100
    observed_coverage.append(coverage)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect calibration')
ax.scatter(nominal_levels, observed_coverage, s=200, alpha=0.7, label='Ensemble')
ax.fill_between([0, 100], [-5, 95], [5, 105], alpha=0.1, color='gray')
ax.set_xlabel('Nominal CI Level (%)')
ax.set_ylabel('Observed Coverage (%)')
ax.set_title('Calibration Curve: Model Confidence Reliability')
ax.set_xlim([40, 100])
ax.set_ylim([40, 100])
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('figure_4_4_calibration_curve.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` — calibration analysis
**Output file:** `results/figures/figure_4_4_calibration_curve.png`

---

### Figure 4.5: Histogram of Percent Difference vs. Historical Mean

**Section:** 4.4.3 (Distribution of Percent Differences)

**Purpose:** Show distribution of predictions relative to historical baseline; visualize frequency of above-average vs. below-average predictions

**Description:**
- X-axis: Percent difference (% above/below historical mean), typically −100% to +100%
- Y-axis: Frequency (number of predictions)
- Histogram bars: Colored by bin:
  - Red: Extreme low (< −40%)
  - Orange: Low (−40% to −20%)
  - Yellow: Below average (−20% to 0%)
  - Light green: Above average (0% to +20%)
  - Green: High (+20% to +40%)
  - Dark green: Extreme high (> +40%)
- Reference line: X=0 (historical mean)
- Statistics: Mean, std dev, skewness annotated

**Suggested format:**
- Matplotlib histogram with color-coded bins
- Size: 8" × 5" (landscape)
- Bins: 20–30 bins

**How to generate:**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-100, 100, 31)
ax.hist(pct_diff, bins=bins, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Historical mean')
ax.set_xlabel('Percent Difference vs. Historical Mean (%)')
ax.set_ylabel('Frequency (# predictions)')
ax.set_title(f'Distribution of Temporal Comparisons\nMean = {pct_diff.mean():.1f}%, Skewness = {scipy.stats.skew(pct_diff):.2f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.savefig('figure_4_5_pct_diff_histogram.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` — temporal comparison section
**Output file:** `results/figures/figure_4_5_pct_diff_histogram.png`

---

### Figure 4.6: Time Series of Predicted Yields vs. Historical Mean (by Crop and Region)

**Section:** 4.4.5 (Temporal Sequence Analysis)

**Purpose:** Show year-to-year variation in predictions; enable visual inspection of whether predictions align with known anomalies

**Description:**
- Subplots: One per crop (Maize, Rice, Cassava, Yam) or one per major region
- X-axis: Year
- Y-axis: Yield (kg/ha)
- Lines:
  - Blue line with dots: Predicted yields (ensemble mean)
  - Shaded band around blue line: ±1.96σ (95% CI)
  - Dashed black line: Historical mean
  - Shaded gray band: Historical ±1σ (reference range)
- Annotations: Known anomalies (drought years, etc.) marked

**Suggested format:**
- Matplotlib subplots or Plotly subplots
- Size: 12" × 8" (landscape, multiple subplots)
- 2×2 or 1×4 grid layout
- Clear legend identifying each line/band

**How to generate:**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

crops = ['Maize', 'Rice', 'Cassava', 'Yam']
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
axes = axes.ravel()

for i, crop in enumerate(crops):
    ax = axes[i]
    subset = predictions[predictions['Crop'] == crop]
    
    # Historical baseline
    hist_mean = historical_data[historical_data['Crop'] == crop]['Yield'].mean()
    hist_std = historical_data[historical_data['Crop'] == crop]['Yield'].std()
    
    # Plot
    ax.plot(subset['Year'], subset['y_pred'], 'b-o', linewidth=2, label='Predicted')
    ax.fill_between(subset['Year'], 
                     subset['y_pred'] - 1.96*subset['y_std'],
                     subset['y_pred'] + 1.96*subset['y_std'],
                     alpha=0.2, color='blue')
    ax.axhline(hist_mean, color='black', linestyle='--', linewidth=2, label='Historical mean')
    ax.fill_between(subset['Year'], hist_mean - hist_std, hist_mean + hist_std, 
                    alpha=0.1, color='gray', label='Hist. ±1σ')
    
    ax.set_title(f'{crop}')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Time Series of Predicted Yields vs. Historical Baseline', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('figure_4_6_timeseries_by_crop.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` — temporal trends section
**Output file:** `results/figures/figure_4_6_timeseries_by_crop.png`

---

### Figure 4.7: Heatmap of Feature Impacts (Features vs. Regions) — LOFO Sensitivity

**Section:** 4.5.4 (Regional Heterogeneity in Drivers)

**Purpose:** Visualize which climate features most influence yields in each region; show regional variation in driver importance

**Description:**
- Rows: 9 climate features (T2M, T2M_MAX, T2M_MIN, TS, PRECTOTCORR, RH2M, QV2M, T2MDEW, T2MWET)
- Columns: 6 regions
- Cell values: Mean LOFO impact (kg/ha) or normalized impact (%)
- Cell color intensity: Darker = higher impact
- Annotations: Numerical value in each cell
- Colorbar: Scale from 0 to max impact

**Suggested format:**
- Seaborn heatmap
- Size: 10" × 6" (landscape)
- Color scheme: viridis or YlGn (yellow=low, green=high)
- Font: Clear feature and region labels

**How to generate:**
```python
import seaborn as sns
import pandas as pd

# Create pivot table: rows=feature, columns=region, values=mean_impact
impact_pivot = lofo_results.groupby(['Feature', 'Region'])['Yield_Impact_kg_ha'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    impact_pivot,
    annot=True,
    fmt='.0f',
    cmap='YlGn',
    cbar_kws={'label': 'Mean LOFO Impact (kg/ha)'},
    ax=ax
)
ax.set_title('Feature Importance by Region (LOFO Sensitivity)')
ax.set_xlabel('Region')
ax.set_ylabel('Climate Feature')
plt.savefig('figure_4_7_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_eval.ipynb` or `notebooks/tcn_mlp_shap.ipynb` — LOFO section
**Output file:** `results/figures/figure_4_7_feature_importance_heatmap.png`

---

### Figure 4.8: Training and Validation Loss Curves for Each Fold

**Section:** 4.6.3 (Cross-Validation Diagnostics)

**Purpose:** Show convergence behavior across 5 folds; verify stable training and absence of overfitting

**Description:**
- Subplots: One per fold (1–5)
- X-axis: Epoch number
- Y-axis: Loss (MSE on log-transformed yields)
- Lines per subplot:
  - Blue line: Training loss (decreasing trend)
  - Orange line: Validation loss (should decrease then plateau)
- Vertical line: Early stopping point (epoch with lowest validation loss)
- Shaded region (optional): Training region after early stopping

**Suggested format:**
- Matplotlib subplots (1×5 or 5 vertically stacked)
- Size: 14" × 6" (very wide landscape) or 8" × 12" (tall portrait)
- All subplots share same y-axis scale for easy comparison

**How to generate:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=True)

for fold in range(5):
    ax = axes[fold]
    history = training_histories[fold]  # Dict with 'train_loss' and 'val_loss' keys
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training loss')
    ax.plot(epochs, history['val_loss'], 'orange', linewidth=2, label='Validation loss')
    
    # Mark early stopping epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    
    ax.set_title(f'Fold {fold + 1}')
    ax.set_xlabel('Epoch')
    if fold == 0:
        ax.set_ylabel('Loss (MSE)')
    ax.grid(alpha=0.3)
    ax.legend()

plt.suptitle('Training Convergence by Fold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figure_4_8_training_loss_curves.png', dpi=300, bbox_inches='tight')
```

**Source:** `notebooks/tcn_mlp_train.ipynb` — training history logs
**Output file:** `results/figures/figure_4_8_training_loss_curves.png`

---

## TABLES

### Table 4.1: Dataset Summary

**Section:** 4.1.1 (Dataset Summary)

**Content:** Overview of dataset characteristics (sample size, geographic coverage, crops, temporal range, missing data)

**Format:** Markdown table
- Row 1: Metric (description of quantity)
- Row 2: Value (actual number or range)
- Row 3: Notes (additional context)

**Source:** Compute from `data/processed_dataset.csv` or dataset documentation
**How to generate:**
```python
import pandas as pd

dataset = pd.read_csv('data/processed_dataset.csv')
print(f"Number of samples: {len(dataset)}")
print(f"Regions: {dataset['Region'].nunique()} ({', '.join(sorted(dataset['Region'].unique()))})")
print(f"Crops: {dataset['Crop'].nunique()} ({', '.join(sorted(dataset['Crop'].unique()))})")
print(f"Years: {dataset['Year'].min()}–{dataset['Year'].max()}")
print(f"Mean yield: {dataset['Yield_kg_per_ha'].mean():.0f} ± {dataset['Yield_kg_per_ha'].std():.0f} kg/ha")
print(f"Missing data: {(dataset.isna().sum() / len(dataset) * 100).mean():.1f}%")
```

**Output file:** `results/tables/table_4_1_dataset_summary.csv` or inline in markdown

---

### Table 4.2: Overall Ensemble Performance Metrics

**Section:** 4.2.1 (Aggregate Ensemble Performance)

**Content:** MAE, RMSE, MAPE, R², MASE on test set

**Format:** Markdown table
- Columns: Metric name, Value, Interpretation

**Source:** `notebooks/tcn_mlp_eval.ipynb` — test set evaluation
**How to generate:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.0f} kg/ha")
print(f"RMSE: {rmse:.0f} kg/ha")
print(f"MAPE: {mape:.1f}%")
print(f"R²: {r2:.3f}")
```

**Output file:** Inline in markdown or `results/tables/table_4_2_ensemble_metrics.csv`

---

### Table 4.3: Ensemble vs. Individual Fold Performance

**Section:** 4.2.2 (Ensemble vs. Single-Model Performance)

**Content:** MAE, RMSE, R² for each of 5 folds and ensemble average

**Format:** Markdown table with 5 rows (folds) + 1 summary row (ensemble)

**Source:** `notebooks/tcn_mlp_eval.ipynb` or `results/ensemble_metrics_mapes_maase_trimmed.csv`

---

### Table 4.4: Per-Crop Performance Breakdown

**Section:** 4.2.3 (Per-Crop Performance)

**Content:** Sample size, MAE, RMSE, MAPE, R² for each crop

**Format:** Markdown table, rows=crops, columns=metrics

**Source:** `results/per_crop_ensemble_mapes_maase_trimmed.csv`

---

### Table 4.5: Per-Region Performance Breakdown

**Section:** 4.2.4 (Per-Region Performance)

**Content:** Sample size, MAE, RMSE, MAPE, R², Mean Yield for each region

**Format:** Markdown table, rows=regions, columns=metrics

**Source:** `results/ensemble_metrics_mapes_maase_trimmed.csv` (aggregated by region)

---

### Table 4.6: Uncertainty Distribution (Low/Medium/High Bands)

**Section:** 4.3.2 (Uncertainty Distribution & Thresholds)

**Content:** Ranges, prediction counts, percentages, model confidence levels

**Format:** Markdown table, rows=bands, columns=range/count/percent/confidence

**Source:** Computed from ensemble predictions and variance

---

### Table 4.7: Feature Importance Ranking (LOFO Sensitivity)

**Section:** 4.5.2 (Overall Feature Importance Ranking)

**Content:** Rank, Feature name, Mean Impact (kg/ha), Median Impact, Std Dev, % of Max Impact

**Format:** Markdown table, sorted by impact (descending)

**Source:** `results/permutation_feature_importance_focused.csv` or equivalent LOFO output

---

### Table 4.8: Per-Crop Feature Importance

**Section:** 4.5.3 (Crop-Specific Drivers)

**Content:** Rows=features, Columns=crops, Cells=impact values

**Format:** Markdown table (wide format)

**Source:** LOFO results disaggregated by crop

---

### Table 4.9: Per-Region Feature Importance

**Section:** 4.5.4 (Regional Heterogeneity in Drivers)

**Content:** Rows=regions, Columns=top 3 drivers with impact values

**Format:** Markdown table

**Source:** LOFO results disaggregated by region

---

### Table 4.10: Per-Fold Performance Summary

**Section:** 4.6.3 (Cross-Validation Diagnostics)

**Content:** Fold number, Training epochs, Final train loss, Final val loss, Test MAE, Std dev of test predictions

**Format:** Markdown table, rows=folds + mean, columns=metrics

**Source:** `notebooks/tcn_mlp_train.ipynb` training logs

---

## SCREENSHOTS & UI CAPTURES

### Screenshot 4.1: Model Input Data (12-Month Climate Sequence Table)

**Section:** 4.1.2 (Climate Features & Preprocessing)

**Purpose:** Show example of input data format for users and stakeholders

**Description:**
- Screenshot of Streamlit data editor showing 12 rows (months) × 9 columns (climate features)
- Filled with representative values (e.g., typical maize season in North-Central region)
- Column headers: T2M, T2M_MAX, T2M_MIN, TS, PRECTOTCORR, RH2M, QV2M, T2MDEW, T2MWET
- Row labels: Jan–Dec

**How to capture:**
1. Open Streamlit app (`streamlit run pages/01_Make_Prediction.py`)
2. Navigate to "Make Prediction" page
3. Enter typical values or load test.csv
4. Screenshot the data editor table
5. Save as PNG

**Output file:** `results/figures/screenshot_4_1_input_data_table.png`

---

### Screenshot 4.2: Prediction Output UI

**Section:** 4.2 & 4.3 (Prediction Results & Uncertainty)

**Purpose:** Show how predictions and confidence are presented to users

**Description:**
- Metrics display: Expected yield range, Model confidence, Best estimate, Model agreement
- Color-coded confidence indicator (green/yellow/red for high/medium/low)
- Prediction chart (bar + error bars)

**How to capture:**
1. Generate a prediction in the Streamlit app
2. Screenshot the metrics and chart section
3. Save as PNG

**Output file:** `results/figures/screenshot_4_2_prediction_output.png`

---

### Screenshot 4.3: Temporal Comparison Results

**Section:** 4.4 (Temporal Comparison vs. Historical Baseline)

**Purpose:** Show how historical comparison is presented (percentile, Z-score, percent difference)

**Description:**
- Text and metrics showing: Historical mean yield, Prediction vs. historical (pct diff, Z-score, percentile)
- Interpretation text (e.g., "Top X% historically")

**How to capture:**
1. Generate a prediction in the Streamlit app
2. Screenshot the "Comparison vs Historical Yield" section
3. Save as PNG

**Output file:** `results/figures/screenshot_4_3_temporal_comparison.png`

---

### Screenshot 4.4: Feature Drivers Table & Chart

**Section:** 4.5 (Drivers & Feature Attribution)

**Purpose:** Show LOFO sensitivity results and driver visualization

**Description:**
- Table showing: Rank, Feature, Parameter, User_Mean, Baseline_Mean, Yield_Impact_kg_ha, Interpretation
- Horizontal bar chart showing feature impacts (blue/red for positive/negative)

**How to capture:**
1. Generate a prediction in the Streamlit app
2. Screenshot the "Drivers Behind This Prediction" section (table + chart)
3. Save as PNG

**Output file:** `results/figures/screenshot_4_4_drivers_table_chart.png`

---

### Screenshot 4.5: Download Buttons & Report Options

**Section:** 4.2 (Report Generation)

**Purpose:** Show how users access reports (Markdown, PDF, ZIP)

**Description:**
- Three download buttons in a row: "Download Markdown report", "Download PDF report", "Download ZIP (MD + PDF)"
- Styled with Streamlit button appearance

**How to capture:**
1. Generate a prediction in the Streamlit app
2. Scroll to the download section
3. Screenshot the three download buttons
4. Save as PNG

**Output file:** `results/figures/screenshot_4_5_download_buttons.png`

---

## SAMPLE REPORT PAGES

### Screenshot 4.6: Sample Markdown Report

**Section:** 4.2 & Appendix (Example output)

**Purpose:** Show what a generated Markdown report looks like

**Description:**
- Markdown-formatted text with sections:
  - Scenario Summary (table)
  - Input Parameters (table)
  - Prediction Context
  - Prediction chart (embedded image)
  - Temporal Comparison
  - Drivers table + chart
  - Confidence explanation
  - Model limitations

**How to generate:**
1. Generate a prediction in the Streamlit app
2. Download the Markdown report (.md file)
3. Open in a Markdown viewer (e.g., VS Code)
4. Screenshot or render to HTML for viewing
5. Save as PNG or PDF

**Output file:** `results/figures/screenshot_4_6_sample_markdown_report.png` or `sample_report.md`

---

### Screenshot 4.7: Sample PDF Report Page 1

**Section:** 4.2 & Appendix (Example output)

**Purpose:** Show what a generated PDF report looks like

**Description:**
- PDF screenshot showing:
  - Title: "Yield Prediction Report"
  - Generated timestamp
  - Scenario Summary table
  - Input Parameters table
  - Prediction Context section
  - Embedded prediction chart

**How to generate:**
1. Generate a prediction in the Streamlit app
2. Download the PDF report
3. Open PDF and screenshot first page
4. Save as PNG

**Output file:** `results/figures/screenshot_4_7_sample_pdf_page1.png`

---

## SUMMARY OF FILES TO GENERATE/COLLECT

| Figure/Table | Type | Format | File Name | Source |
|---|---|---|---|---|
| Figure 4.1 | Plot | PNG | figure_4_1_obs_vs_pred.png | notebooks/tcn_mlp_eval.ipynb |
| Figure 4.2 | Heatmap | PNG | figure_4_2_regional_heatmap.png | notebooks/tcn_mlp_eval.ipynb |
| Figure 4.3 | Histogram | PNG | figure_4_3_uncertainty_histogram.png | notebooks/tcn_mlp_eval.ipynb |
| Figure 4.4 | Plot | PNG | figure_4_4_calibration_curve.png | notebooks/tcn_mlp_eval.ipynb |
| Figure 4.5 | Histogram | PNG | figure_4_5_pct_diff_histogram.png | notebooks/tcn_mlp_eval.ipynb |
| Figure 4.6 | Time series | PNG | figure_4_6_timeseries_by_crop.png | notebooks/tcn_mlp_eval.ipynb |
| Figure 4.7 | Heatmap | PNG | figure_4_7_feature_importance_heatmap.png | notebooks/tcn_mlp_shap.ipynb |
| Figure 4.8 | Multi-plot | PNG | figure_4_8_training_loss_curves.png | notebooks/tcn_mlp_train.ipynb |
| Table 4.1 | CSV/Markdown | PNG or .csv | table_4_1_dataset_summary | data/processed_dataset.csv |
| Table 4.2 | Markdown | PNG or .csv | table_4_2_ensemble_metrics | eval results |
| Table 4.3–4.10 | Markdown | PNG or .csv | table_4_X_* | eval results |
| Screenshot 4.1 | Data table | PNG | screenshot_4_1_input_data_table.png | Streamlit app |
| Screenshot 4.2 | UI output | PNG | screenshot_4_2_prediction_output.png | Streamlit app |
| Screenshot 4.3 | UI output | PNG | screenshot_4_3_temporal_comparison.png | Streamlit app |
| Screenshot 4.4 | UI output | PNG | screenshot_4_4_drivers_table_chart.png | Streamlit app |
| Screenshot 4.5 | UI output | PNG | screenshot_4_5_download_buttons.png | Streamlit app |
| Screenshot 4.6 | Report | PNG/MD | screenshot_4_6_sample_markdown_report.png | Generated report |
| Screenshot 4.7 | Report | PNG | screenshot_4_7_sample_pdf_page1.png | Generated PDF |

---

## WORKFLOW FOR INTEGRATION

1. **Generate figures from notebooks:**
   ```bash
   # Run evaluation notebook to generate Figures 4.1–4.7
   jupyter nbconvert --to notebook --execute notebooks/tcn_mlp_eval.ipynb
   
   # Run training notebook to extract Figure 4.8
   jupyter nbconvert --to notebook --execute notebooks/tcn_mlp_train.ipynb
   ```

2. **Capture Streamlit screenshots:**
   ```bash
   # Start Streamlit app
   streamlit run pages/01_Make_Prediction.py
   
   # Use browser screenshot tool or screen capture utility
   # For Windows: Snipping Tool, Shift+Windows+S
   # For Mac: Cmd+Shift+4
   # For Linux: gnome-screenshot or similar
   ```

3. **Organize files:**
   ```
   results/
   ├── figures/
   │   ├── figure_4_1_obs_vs_pred.png
   │   ├── figure_4_2_regional_heatmap.png
   │   ├── ...
   │   └── screenshot_4_7_sample_pdf_page1.png
   ├── tables/
   │   ├── table_4_1_dataset_summary.csv
   │   ├── table_4_2_ensemble_metrics.csv
   │   └── ...
   └── sample_reports/
       ├── sample_prediction_report.md
       └── sample_prediction_report.pdf
   ```

4. **Link in Chapter 4 markdown:**
   - Update placeholders in `thesis_chapter_4_detailed.md`
   - Replace `**[Figure 4.X]**` with `![Figure 4.X description](../results/figures/figure_4_X_filename.png)`
   - Replace `**[Table 4.X]**` with embedded markdown table or reference to CSV file

---

## QUALITY CHECKLIST

Before including in final thesis:

- [ ] All figures have clear titles and axis labels
- [ ] All figures use consistent color schemes and fonts
- [ ] All figures include legends and/or captions
- [ ] All figures are high-resolution (300 DPI for print)
- [ ] All tables are readable at book size (10pt+ font)
- [ ] All tables have clear row/column headers
- [ ] All screenshots include relevant context (no clutter, no sensitive info)
- [ ] All file names follow naming convention (figure_4_X_*, table_4_X_*, screenshot_4_X_*)
- [ ] All figures referenced in text; all figures in appendix or main text
- [ ] Captions are descriptive and match figure content

---

**End of Screenshots & Images Reference Guide**

*Use this document as a checklist to ensure all visual elements for Chapter 4 are collected, formatted, and integrated into the final thesis.*
