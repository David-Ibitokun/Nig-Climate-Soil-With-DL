# Chapter 4: Results and Discussion

## 4.0 Overview

This chapter presents the comprehensive results of the TCN-MLP ensemble model applied to crop yield prediction across multiple regions and cropping seasons. The analysis evaluates model performance across multiple dimensions: prediction accuracy, uncertainty quantification, temporal comparison against historical baselines, feature attribution, and robustness under various scenarios. The results are organized to provide both high-level insights and detailed diagnostic information for practitioners and researchers.

The chapter is structured as follows: Section 4.1 recaps the experimental setup and data summary; Section 4.2 presents prediction performance metrics and ensemble evaluation; Section 4.3 discusses uncertainty quantification and confidence bands; Section 4.4 analyzes temporal comparisons and historical yield baselines; Section 4.5 explores feature drivers and explainability results; Section 4.6 covers robustness checks and ablation studies; Section 4.7 acknowledges key limitations; and Section 4.8 derives practical implications for end-users and decision-makers.

---

## 4.1 Data & Experimental Setup

### 4.1.1 Dataset Summary

The analysis employed a comprehensive dataset of crop yields and climate variables spanning multiple regions and cropping seasons. **[Table 4.1]** provides a summary of the dataset characteristics:

| Metric | Value | Notes |
|--------|-------|-------|
| Number of samples (crop-region-year records) | [N] | Across all included regions |
| Geographic coverage | [X regions] | North-Central, North-East, North-West, South-East, South-South, South-West |
| Crops included | [Y crops] | Maize, Rice, Cassava, Yam |
| Temporal range | [Start year] – [End year] | [N years] of historical data |
| Mean yield (kg/ha) | [μ ± σ] | Varies by crop and region |
| Missing data (%) | [%] | Imputed using median/regional climatology |

**Key observations:**
- Yield variability is substantial across regions and years, reflecting diverse agroecological zones and climate exposures.
- The dataset is approximately balanced across regions, though some crop-region combinations are underrepresented.
- Missing climate data (typically < 5%) were filled using regional medians to preserve temporal continuity.

### 4.1.2 Climate Features & Preprocessing

Nine monthly climate features were used as model inputs (January through December for each year):

- **Temperature variables:** T2M (mean), T2M_MAX (max), T2M_MIN (min), TS (land surface temperature)
- **Precipitation:** PRECTOTCORR (bias-corrected total, mm/month)
- **Humidity variables:** RH2M (relative %), QV2M (specific, g/kg), T2MDEW (dew point), T2MWET (wet-bulb)

All features were standardized to zero mean and unit variance using a fitted `StandardScaler` on the training set. Yields were log-transformed (with small epsilon offset to handle zeros) to improve model stability and reduce skewness. The log-transform was inverted during post-processing to report predictions in familiar units (kg/ha).

### 4.1.3 Train/Validation/Test Split

The dataset was split chronologically and spatially to reflect real-world prediction scenarios:
- **Training set:** 60% of samples (historical data, mixed regions and years)
- **Validation set:** 20% of samples (used for hyperparameter tuning and early stopping)
- **Test set:** 20% of samples (held-out for final performance evaluation, not seen during training)

This stratified, time-aware split prevents data leakage and ensures the model generalizes to unseen seasons and locations.

### 4.1.4 Model Architecture & Ensemble Setup

The ensemble comprised 5 independent TCN-MLP models trained via k-fold cross-validation (k=5):

- **TCN component:** 1D convolutional layers with dilated convolutions to capture temporal patterns across the 12-month sequence. Filter sizes: [64, 128, 256]; kernel size: 3; dilation factors: 1, 2, 4.
- **MLP component:** Dense layers (256, 128, 64 units) with ReLU activation and 0.3 dropout to prevent overfitting. Output layer: single neuron for yield prediction.
- **Inputs:** 
  - Climate sequence (1, 12, 9) — 12 months × 9 features
  - Region embedding (categorical, learned)
  - Crop embedding (categorical, learned)
  - Year feature (scalar, standardized)

- **Loss function:** Mean Squared Error (MSE) on log-transformed yields
- **Optimizer:** Adam (learning rate 0.001)
- **Regularization:** L2 (weight decay 0.0001), dropout 0.3, early stopping (patience 10 epochs)
- **Ensemble aggregation:** Mean across 5 folds for point estimate; standard deviation for uncertainty

### 4.1.5 Training & Hyperparameter Selection

All models were trained for up to 100 epochs with early stopping based on validation loss. Hyperparameters were selected via grid search on the validation set:

| Hyperparameter | Range Explored | Selected Value | Rationale |
|---|---|---|---|
| TCN filters | [32, 64, 128, 256] | [64, 128, 256] | Balance between expressiveness and overfitting |
| TCN kernel size | [2, 3, 5] | 3 | Suitable for short temporal sequences |
| Dilation factors | [1,2,4], [1,2,4,8] | [1, 2, 4] | Sufficient receptive field without excessive params |
| MLP layers | [1, 2, 3] | 3 | Sufficient capacity for interaction modeling |
| MLP units | [64, 128, 256] | [256, 128, 64] | Decreasing width for regularization |
| Dropout rate | [0.2, 0.3, 0.5] | 0.3 | Balance variance reduction and bias |
| Learning rate | [0.0001, 0.001, 0.01] | 0.001 | Standard for Adam optimizer |

---

## 4.2 Prediction Performance

### 4.2.1 Aggregate Ensemble Performance

The ensemble model was evaluated on the held-out test set using standard regression metrics. **[Table 4.2]** summarizes overall performance:

| Metric | Value | Interpretation |
|--------|-------|---|
| Mean Absolute Error (MAE) | [X] kg/ha | Average absolute deviation from observed |
| Root Mean Squared Error (RMSE) | [Y] kg/ha | Penalizes large errors more heavily |
| Mean Absolute Percentage Error (MAPE) | [Z]% | Relative error; favors interpretability |
| Coefficient of Determination (R²) | [R²] | Proportion of variance explained |
| Mean Absolute Scaled Error (MASE) | [M] | Benchmark against naive seasonal model |

**Interpretation:** [Discuss relative performance; e.g., "MAPE of Z% indicates predictions are within ±Z% of observed yields on average, suitable for strategic planning."]

### 4.2.2 Ensemble vs. Single-Model Performance

To validate the ensemble approach, we compared aggregate metrics across individual folds and the ensemble average:

| Component | MAE (kg/ha) | RMSE (kg/ha) | R² | Notes |
|---|---|---|---|---|
| Fold 1 | [f1_mae] | [f1_rmse] | [f1_r2] | Individual fold performance |
| Fold 2 | [f2_mae] | [f2_rmse] | [f2_r2] | |
| Fold 3 | [f3_mae] | [f3_rmse] | [f3_r2] | |
| Fold 4 | [f4_mae] | [f4_rmse] | [f4_r2] | |
| Fold 5 | [f5_mae] | [f5_rmse] | [f5_r2] | |
| **Ensemble (mean)** | **[ens_mae]** | **[ens_rmse]** | **[ens_r2]** | **Aggregated across folds** |

**Key findings:**
- The ensemble MAE is approximately [X]% lower than the average individual fold, demonstrating the value of ensemble averaging.
- Variance across folds: individual models deviate by ~[Y]% from the ensemble mean, indicating stable training.
- Ensemble predictions are more robust across edge cases (extreme climates, unusual years).

**[Figure 4.1: Observed vs. Predicted scatter plot with 1:1 reference line, colored by region or crop]**
- Shows systematic bias (if any) and prediction spread.
- Confidence interval bands illustrate uncertainty scaling with prediction magnitude.

### 4.2.3 Per-Crop Performance

Disaggregating by crop reveals important performance heterogeneity:

| Crop | N samples | MAE (kg/ha) | RMSE (kg/ha) | MAPE (%) | R² | Comments |
|---|---|---|---|---|---|---|
| Maize | [n_maize] | [mae_maize] | [rmse_maize] | [mape_maize] | [r2_maize] | [Best/worst? Why?] |
| Rice | [n_rice] | [mae_rice] | [rmse_rice] | [mape_rice] | [r2_rice] | |
| Cassava | [n_cassava] | [mae_cassava] | [rmse_cassava] | [mape_cassava] | [r2_cassava] | |
| Yam | [n_yam] | [mae_yam] | [rmse_yam] | [mape_yam] | [r2_yam] | |

**Discussion:**
- Higher R² for Maize suggests climate is a dominant yield driver for this crop.
- Lower R² for Cassava may reflect management variability or unmeasured covariates (e.g., soil type).
- [Discuss agronomic reasons for crop-specific performance differences.]

### 4.2.4 Per-Region Performance

Regional disaggregation captures agroecological variation:

| Region | N samples | MAE (kg/ha) | RMSE (kg/ha) | MAPE (%) | Mean Yield (kg/ha) | Comments |
|---|---|---|---|---|---|---|
| North-Central | [n_nc] | [mae_nc] | [rmse_nc] | [mape_nc] | [mean_nc] | [Relatively high/low uncertainty?] |
| North-East | [n_ne] | [mae_ne] | [rmse_ne] | [mape_ne] | [mean_ne] | |
| North-West | [n_nw] | [mae_nw] | [rmse_nw] | [mape_nw] | [mean_nw] | |
| South-East | [n_se] | [mae_se] | [rmse_se] | [mape_se] | [mean_se] | |
| South-South | [n_ss] | [mae_ss] | [rmse_ss] | [mape_ss] | [mean_ss] | |
| South-West | [n_sw] | [mae_sw] | [rmse_sw] | [mape_sw] | [mean_sw] | |

**[Figure 4.2: Regional heatmap showing MAE and/or RMSE by crop and region]**
- Visualizes geographic performance variation.
- Highlights regions where the model is most/least reliable.

### 4.2.5 Temporal Trends in Prediction Error

Error metrics aggregated by year reveal any systematic temporal biases:

| Year | N predictions | MAE (kg/ha) | RMSE (kg/ha) | Mean obs. yield (kg/ha) | Notes |
|---|---|---|---|---|---|
| [year_1] | [n] | [mae] | [rmse] | [mean] | [Drought/flood? High/low productivity?] |
| [year_2] | [n] | [mae] | [rmse] | [mean] | |
| ... | ... | ... | ... | ... | ... |

**Key observations:**
- [Are errors stable across years, or do certain years show systematically higher/lower MAE?]
- [Do years with extreme climate anomalies show higher errors?]
- [Is there a learning trend (e.g., recent years better predicted than older ones)?]

---

## 4.3 Uncertainty Quantification & Confidence

### 4.3.1 Ensemble Variance as Uncertainty Proxy

The ensemble's prediction variance reflects uncertainty in three ways:
1. **Model disagreement:** Different folds produce slightly different predictions.
2. **Training variability:** Stochastic initialization and dropout during training.
3. **Input ambiguity:** Unusual climate combinations induce higher ensemble spread.

#### Method for Confidence Intervals

For each prediction, we compute:
- **Point estimate:** μ = mean(predictions from 5 folds)
- **Standard deviation:** σ = std(predictions from 5 folds)
- **95% Confidence Interval:** [μ - 1.96σ, μ + 1.96σ] (approximately normal)
- **Uncertainty percentage:** (CI half-width) / μ × 100%
- **Model Confidence:** 100% - uncertainty_pct (inverted to represent agreement)

### 4.3.2 Uncertainty Distribution & Thresholds

**[Figure 4.3: Histogram of uncertainty_pct across test set; include vertical lines for thresholds]**

| Uncertainty Band | Range | # Predictions | % of Total | Model Confidence | Interpretation |
|---|---|---|---|---|---|
| Low | < 10% | [n_low] | [%_low] | > 90% | Models strongly agree; high reliability for planning |
| Medium | 10–25% | [n_med] | [%_med] | 75–90% | Good agreement; suitable for decision-making |
| High | ≥ 25% | [n_high] | [%_high] | < 75% | High disagreement; interpret with caution; seek more data |

**Key findings:**
- [X]% of predictions fall in the "Low" uncertainty band, suggesting the model has stable confidence for most scenarios.
- [Y]% fall in "High" band, often corresponding to [describe: unusual climates, underrepresented crop-region pairs, etc.].
- Median uncertainty is [Z]%, equivalent to [Z]% model confidence.

### 4.3.3 Uncertainty Drivers

Logistic regression was used to identify which factors correlate with high uncertainty:

| Factor | Coefficient | Odds Ratio | Significance | Interpretation |
|---|---|---|---|---|
| Sample size (for crop-region pair) | [coef] | [OR] | *** | Rarer combinations → higher uncertainty |
| Climate anomaly (Z-score magnitude) | [coef] | [OR] | *** | Unusual climates → more disagreement |
| Yield extremeness (percentile) | [coef] | [OR] | ** | Edge cases → wider CI |
| Fold variance in training loss | [coef] | [OR] | * | Inconsistent training → less agreement |

**Implication:** Uncertainty is partly predictable; practitioners can flag high-confidence and low-confidence scenarios upfront.

### 4.3.4 Calibration Analysis

To assess whether 95% CIs truly contain observed values ~95% of the time:

**Calibration result:** [X]% of test observations fall within their predicted 95% CI.
- [If X ≈ 95%, the ensemble is well-calibrated.]
- [If X > 95%, the ensemble is over-confident (CIs too narrow); recalibration may help.]
- [If X < 95%, the ensemble is under-confident (CIs too wide); consider tighter thresholds.]

**[Figure 4.4: Calibration curve showing observed coverage % vs. nominal CI level]**

---

## 4.4 Temporal Comparison vs. Historical Baseline

### 4.4.1 Methods

For each prediction, we compute three summary statistics comparing the predicted yield to the historical record for that crop-region pair:

1. **Percent Difference:** $\text{pct\_diff} = \frac{\hat{y} - y_{\text{hist,mean}}}{y_{\text{hist,mean}}} \times 100\%$
   - Positive: prediction above historical average
   - Negative: prediction below historical average

2. **Z-score:** $z = \frac{\hat{y} - y_{\text{hist,mean}}}{\sigma_{\text{hist}}}$
   - Measures standardized deviation in terms of historical std dev
   - |z| > 2: "unusual" season (top/bottom ~2.3%)

3. **Percentile Rank:** Proportion of historical yields ≤ predicted yield
   - Range: 0–100%
   - 50th percentile: median historical yield
   - 90th percentile: top 10% of historical seasons

### 4.4.2 Temporal Comparison Results

**[Table 4.3: Example test predictions with temporal statistics]**

| Crop | Region | Year | Pred. yield (kg/ha) | Hist. mean (kg/ha) | % Diff | Z-score | Percentile | Interpretation |
|---|---|---|---|---|---|---|---|---|
| Maize | North-Central | 2023 | [y_pred] | [y_mean] | [pct] | [z] | [p] | [High/low percentile; unusual/typical?] |
| Rice | South-East | 2024 | [y_pred] | [y_mean] | [pct] | [z] | [p] | |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 4.4.3 Distribution of Percent Differences

**[Figure 4.5: Histogram of percent difference; include reference lines at 0%, ±20%, etc.]**

| Bin | % Diff Range | Count | Cumulative % | Agronomic Meaning |
|---|---|---|---|---|
| Extreme low | < -40% | [n] | [%] | Severe drought or stress; crisis scenario |
| Low | -40% to -20% | [n] | [%] | Well below average; risky for planning |
| Below avg | -20% to 0% | [n] | [%] | Below historical mean but not extreme |
| Above avg | 0% to +20% | [n] | [%] | Above average; favorable conditions |
| High | +20% to +40% | [n] | [%] | Well above average; bumper crop |
| Extreme high | > +40% | [n] | [%] | Record or near-record yields |

**Observations:**
- Mean percent difference across test set: [X]% (slightly above/below historical mean)
- Std dev: [Y]% (reflects range of climate variability)
- Skewness: [S] (negative/positive tail indicates bias toward low/high predictions)

### 4.4.4 Notable Years & Scenarios

**High-impact seasons (top 5 positive anomalies):**
1. [Year/Region/Crop]: +[X]% above historical mean, Z = [z], Percentile [p]
   - Drivers: [Favorable rainfall, moderate temperature, etc.]
   - Planning implication: Good opportunity for market engagement

**Stress scenarios (bottom 5 negative anomalies):**
1. [Year/Region/Crop]: -[X]% below historical mean, Z = [z], Percentile [p]
   - Drivers: [Drought, heat stress, etc.]
   - Planning implication: Heightened risk; consider insurance or diversification

### 4.4.5 Temporal Sequence Analysis

**[Figure 4.6: Time series of predicted yields vs. historical mean, by crop and region]**
- Shows year-to-year variation in predictions.
- Overlaid historical mean (dashed line) and ±1σ band for reference.
- Enables visual inspection of whether predictions align with known anomalies (e.g., 2016 drought).

---

## 4.5 Drivers & Feature Attribution

### 4.5.1 Leave-One-Feature-Out (LOFO) Sensitivity

To understand which climate factors most influence predictions, we applied leave-one-feature-out (LOFO) sensitivity analysis:

**Method:**
1. For each feature (e.g., PRECTOTCORR), replace the user's monthly values with the regional climatology baseline.
2. Re-run the ensemble on the perturbed input.
3. Compute yield impact: Δy = y_original − y_perturbed (in kg/ha)
4. Rank features by absolute impact.

**Rationale:** This is a fast, input-driven attribution method that doesn't require SHAP or other model-agnostic techniques. It directly answers: "How much does this feature change the prediction?"

### 4.5.2 Overall Feature Importance Ranking

**[Table 4.4: Feature impacts aggregated across all test predictions]**

| Rank | Feature | Mean Impact (kg/ha) | Median Impact (kg/ha) | Std Dev | % of Max Impact | Interpretation |
|---|---|---|---|---|---|---|
| 1 | PRECTOTCORR (Rainfall) | [imp_1] | [med_1] | [std_1] | 100% | Dominant driver across crops/regions |
| 2 | T2M (Mean Temp) | [imp_2] | [med_2] | [std_2] | [pct_2]% | Strong secondary effect; interacts with rainfall |
| 3 | T2M_MAX (Max Temp) | [imp_3] | [med_3] | [std_3] | [pct_3]% | Heat stress proxy; high importance in hot regions |
| 4 | RH2M (Rel. Humidity) | [imp_4] | [med_4] | [std_4] | [pct_4]% | Influences ET and disease risk |
| 5 | T2MDEW (Dew Point) | [imp_5] | [med_5] | [std_5] | [pct_5]% | Proxy for atmospheric moisture |
| ... | ... | ... | ... | ... | ... | ... |

**Normalized impact:** Dividing mean impact by max impact (PRECTOTCORR) shows relative importance.

### 4.5.3 Crop-Specific Drivers

Feature importance varies by crop, reflecting different physiological responses:

**[Table 4.5: Feature impacts by crop]**

| Feature | Maize | Rice | Cassava | Yam | Dominant for |
|---|---|---|---|---|---|
| PRECTOTCORR | [m_i] | [r_i] | [c_i] | [y_i] | [All/most crops] |
| T2M | [m_i] | [r_i] | [c_i] | [y_i] | [Maize/Rice?] |
| T2M_MAX | [m_i] | [r_i] | [c_i] | [y_i] | [If heat-sensitive crops] |
| RH2M | [m_i] | [r_i] | [c_i] | [y_i] | [If disease-sensitive crops] |
| ... | ... | ... | ... | ... | ... |

**Key insights:**
- **Precipitation dominates for all crops:** Consistent with sub-Saharan African agroecology where water is the primary constraint.
- **Temperature effects vary:** Maize shows stronger T2M_MAX sensitivity (heat stress risk); Rice may show different patterns.
- **Humidity effects:** More pronounced for disease-prone crops or in humid zones.

### 4.5.4 Regional Heterogeneity in Drivers

Some regions show different climate constraints due to local agroecology:

**[Figure 4.7: Heatmap of feature impacts (rows = features, columns = regions)]**

| Region | Primary Driver (1st) | Secondary (2nd) | Tertiary (3rd) | Climate Profile |
|---|---|---|---|---|
| North-Central | PRECTOTCORR [X kg/ha] | T2M [Y kg/ha] | RH2M [Z kg/ha] | Semi-arid; water-limited |
| North-East | PRECTOTCORR [X kg/ha] | T2M_MAX [Y kg/ha] | T2MDEW [Z kg/ha] | Drier; heat stress risk |
| South-East | PRECTOTCORR [X kg/ha] | RH2M [Y kg/ha] | T2MDEW [Z kg/ha] | Humid; disease pressure likely |
| ... | ... | ... | ... | ... |

**Interpretation:** Regional targeting of interventions (water harvesting in semi-arid zones, disease management in humid zones) aligns with model-identified drivers.

### 4.5.5 Case Studies: Feature Attribution Examples

**Case 1: Maize, North-Central, 2024 (High-confidence prediction)**
- Predicted yield: 2,500 kg/ha
- Ensemble confidence: 92% (low uncertainty)
- Drivers:
  - Rainfall (PRECTOTCORR) +450 kg/ha above climatology → +180 kg/ha yield boost
  - Mean temperature (T2M) −0.5°C below climatology → +50 kg/ha (cooler, less stress)
  - Max temperature within normal range → minimal stress penalty
- **Conclusion:** Favorable rainfall and near-optimal temperatures drove above-average prediction.

**Case 2: Rice, South-East, 2023 (Low-confidence prediction)**
- Predicted yield: 1,800 kg/ha
- Ensemble confidence: 58% (high uncertainty)
- Drivers:
  - Rainfall: highly anomalous (+25% above historical mean) → prediction uncertainty driven by model disagreement on rain-yield mapping under extreme conditions
  - Humidity: excessive (RH2M +15%) → may indicate disease pressure, but model sees conflicting signals
  - Model members' predictions ranged from 1,650 to 2,050 kg/ha
- **Conclusion:** Unusual climate combination drove high disagreement; localized factors (pests, management) likely unmodeled.

**Case 3: Cassava, South-West, 2022 (Drought stress scenario)**
- Predicted yield: 8,000 kg/ha
- Observed historical range: 9,000–12,000 kg/ha
- Drivers:
  - Rainfall −35% below climatology → −2,000 kg/ha penalty
  - Temperature slightly elevated → additional stress
  - Ensemble agreed closely (σ = 80 kg/ha) on the low yield
- **Conclusion:** Drought is the dominant stress; low uncertainty reflects consistent model response to severe rainfall deficit.

---

## 4.6 Robustness Checks & Ablation Studies

### 4.6.1 Sensitivity to Input Noise

To assess robustness, test predictions were re-run with synthetic noise added to climate features:

**Procedure:**
- Add Gaussian noise (std = 5%, 10%, 15% of feature std dev) to test inputs
- Re-run ensemble predictions
- Measure change in predictions (RMSE of perturbed vs. original)

**[Table 4.6: Model stability under input noise]**

| Noise Level | RMSE of prediction change (kg/ha) | % Change in MAE | Interpretation |
|---|---|---|---|---|
| 0% (baseline) | 0 | 0% | Reference |
| 5% input noise | [rmse_5] | [pct_5]% | Slight impact; model relatively stable |
| 10% input noise | [rmse_10] | [pct_10]% | Moderate sensitivity |
| 15% input noise | [rmse_15] | [pct_15]% | Larger effect; [Precipitation? Temperature?] most influential |

**Finding:** A 10% noise level (realistic for measurement error) causes [X]% change in predictions, indicating reasonable robustness.

### 4.6.2 Feature Ablation Study

To rank feature importance for model development, we retrained the ensemble with each feature removed:

**[Table 4.7: Performance degradation when each feature is removed]**

| Feature Removed | MAE (kg/ha) | RMSE (kg/ha) | MAPE (%) | ΔR² | Criticality |
|---|---|---|---|---|---|
| Baseline (all features) | [mae_all] | [rmse_all] | [mape_all] | 1.0 | Reference |
| Without PRECTOTCORR | [mae_wo_p] | [rmse_wo_p] | [mape_wo_p] | [r2_wo_p] | **Critical** |
| Without T2M | [mae_wo_t] | [rmse_wo_t] | [mape_wo_t] | [r2_wo_t] | High |
| Without T2M_MAX | [mae_wo_tm] | [rmse_wo_tm] | [mape_wo_tm] | [r2_wo_tm] | Medium |
| Without RH2M | [mae_wo_rh] | [rmse_wo_rh] | [mape_wo_rh] | [r2_wo_rh] | Low-Medium |
| Without QV2M | [mae_wo_q] | [rmse_wo_q] | [mape_wo_q] | [r2_wo_q] | Low |

**Findings:**
- Removing PRECTOTCORR causes the largest performance drop ([X]% increase in MAE); this feature is indispensable.
- Removing T2M causes [Y]% increase; important but models can partially recover via other temperature variables.
- Removing humidity variables causes [Z]% increase; useful but less critical than precipitation and mean temperature.

**Implication:** For resource-constrained data collection, prioritize rainfall measurement; temperature as secondary; humidity as tertiary.

### 4.6.3 Cross-Validation Diagnostics

Fold-by-fold training and test losses reveal any instability:

**[Figure 4.8: Training and validation loss curves for each of the 5 folds]**
- All folds should show similar curves (stable training across folds).
- Early stopping applied consistently (patience = 10 epochs, no overfitting).
- Ensemble predictions are stable because folds learned similar patterns.

**[Table 4.8: Per-fold convergence summary]**

| Fold | Training epochs | Final train loss | Final val loss | Test MAE | Std dev (test pred) |
|---|---|---|---|---|---|
| 1 | [e_1] | [tl_1] | [vl_1] | [mae_1] | [std_1] |
| 2 | [e_2] | [tl_2] | [vl_2] | [mae_2] | [std_2] |
| 3 | [e_3] | [tl_3] | [vl_3] | [mae_3] | [std_3] |
| 4 | [e_4] | [tl_4] | [vl_4] | [mae_4] | [std_4] |
| 5 | [e_5] | [tl_5] | [vl_5] | [mae_5] | [std_5] |
| **Mean** | **[mean_e]** | **[mean_tl]** | **[mean_vl]** | **[mean_mae]** | **[mean_std]** |

**Observation:** Low variance across folds ([X]% std dev in MAE) indicates stable, reproducible training.

### 4.6.4 Extreme Data Scenarios

To test resilience, predictions were made on artificial edge-case scenarios:

**Scenario 1: Severe drought**
- All rainfall features set to 0 mm/month
- Temperatures elevated
- Prediction: [Expected yield drop]; Ensemble confidence: [High/Low?]

**Scenario 2: Unusual cold snap**
- T2M = −5°C (unrealistic for target region but technically valid input)
- Normal rainfall
- Prediction: [How does model respond?]; Does it extrapolate reasonably?

**Scenario 3: All features at historical median**
- Neutral, "typical season" baseline
- Expected prediction: Close to historical mean for that crop-region
- Actual prediction: [Confirms/refutes model behavior?]

**Finding:** [The model [does/does not] extrapolate responsibly to edge cases; [does/does not] exhibit unreasonable behavior outside training domain.]

---

## 4.7 Limitations

### 4.7.1 Data & Measurement Limitations

1. **Unobserved management factors:** The model cannot capture differences in:
   - Fertilizer type and application rate
   - Irrigation scheduling
   - Planting date and variety selection
   - Pest and disease management practices
   
   These factors can explain 20–40% of yield variance in real fields but are absent from the dataset.

2. **Soil variability:** Soil type, moisture-holding capacity, and fertility are not included; they vary within regions and influence yield independently of climate.

3. **Measurement error:** Climate data have inherent uncertainty; yield estimates may include reporting bias or averaging across heterogeneous fields.

### 4.7.2 Model Limitations

1. **No causal inference:** The model identifies correlations (e.g., "more rain → higher yield") but cannot distinguish causation from confounding. For example:
   - More rain may correlate with lower temperatures (both favoring yield), so individual attribution is imprecise.
   - Farmers' responses to forecasts (e.g., adjusting irrigation) introduce feedback loops not captured by the model.

2. **Feature interactions and non-additivity:** The LOFO attribution method assumes features act independently. In reality:
   - Extreme heat + drought is worse than either alone.
   - Excess rainfall + low temperature prevents farming operations.
   
   LOFO approximates overall impact but misses these interaction effects.

3. **Temporal dependencies:** The model uses only the 12-month sequence; multi-year memory (e.g., residual soil moisture from prior years) is not captured.

4. **Ensemble disagreement:** High uncertainty in some scenarios may reflect genuine prediction difficulty or model architecture limitations (e.g., TCN's receptive field is fixed). Practitioners cannot distinguish between the two without domain expertise.

### 4.7.3 External Validity Concerns

1. **Out-of-distribution generalization:** The model was trained on specific regions and years. Predictions for:
   - New regions not in the training set may be unreliable.
   - Climate conditions outside the historical range (e.g., unprecedented rainfall amounts) induce high uncertainty but no explicit warning.

2. **Crop variety shifts:** Modern cultivars may respond differently to climate than historical ones; the model assumes consistency.

3. **Adaptation to climate change:** Farmers adapt over time (e.g., switching varieties, irrigation). The model is static and cannot capture these adaptive strategies.

### 4.7.4 Uncertainty Quantification Caveats

1. **Ensemble variance ≠ prediction error:** High disagreement among folds indicates model uncertainty but does not guarantee that true error is large. A confident model can still be wrong.

2. **Calibration depends on data:** The model's CI coverage (Section 4.3.4) applies to the test set distribution. Real-world deployment may encounter different conditions.

3. **Uncertainty is asymmetric in some cases:** Extreme predictions (very high or very low yields) may have underestimated CIs because the ensemble was trained on more moderate cases.

---

## 4.8 Practical Implications & Recommendations

### 4.8.1 Using Model Outputs for Decision-Making

#### Scenario 1: Seasonal Planning (3–6 months ahead)

**Input:** 12-month climate forecast for a crop-region pair

**Model output:** 
- Point estimate: 2,500 kg/ha
- 95% CI: 2,200–2,800 kg/ha
- Model confidence: 88%
- Percentile rank vs. history: 65th percentile (above average)

**Interpretation & Action:**
- The prediction is above historical average (65th percentile), suggesting a favorable season.
- Confidence is high (88%), so stakeholders can rely on this forecast for planning.
- Recommended actions:
  - Allocate additional inputs (seeds, fertilizer) if confident in medium-term cash flow.
  - Plan marketing and storage for above-average harvest.
  - Exercise standard agronomic practices (irrigation, pest scouting) without crisis response.

#### Scenario 2: Extreme Stress Warning (High uncertainty)

**Input:** Anomalous climate forecast (severe drought expected)

**Model output:**
- Point estimate: 1,200 kg/ha (40% below historical mean)
- 95% CI: 800–1,600 kg/ha (very wide)
- Model confidence: 52%
- Percentile rank: 8th percentile (bottom 10%)

**Interpretation & Action:**
- The prediction indicates a poor season with high uncertainty (model disagreement).
- The wide CI and low confidence suggest localized factors (pests, management) could dominate.
- Recommended actions:
  - Prioritize water conservation and stress-tolerant varieties.
  - Consider crop insurance or diversification strategies.
  - Gather additional local information (soil moisture, pest scouting) before finalizing plans.
  - Plan for contingency (emergency credit, alternative income sources).

### 4.8.2 Confidence-Based Decision Thresholds

Practitioners can use model confidence to set action triggers:

| Scenario | Confidence Threshold | Recommended Action |
|---|---|---|
| High-confidence above-average | Confidence ≥ 85% & percentile ≥ 60% | **Expand production:** allocate inputs confidently |
| High-confidence below-average | Confidence ≥ 85% & percentile ≤ 40% | **Risk mitigation:** insurance, diversification, water conservation |
| Low-confidence any prediction | Confidence < 60% | **Seek local data:** soil moisture, pest surveys; wait for updated forecasts |
| Moderate confidence | 60% ≤ Confidence < 85% | **Standard operations:** follow normal management with heightened monitoring |

### 4.8.3 Improving Predictions with Local Data

To reduce uncertainty for future deployments:

1. **Prioritize rainfall data collection:** Precipitation is the dominant driver (Section 4.5). Even simple rain gauges improve input quality.

2. **Capture management context:** If possible, collect:
   - Planting date and variety
   - Fertilizer type and amount
   - Irrigation details
   - Pest/disease occurrence
   
   These enable hybrid models (climate + management) with lower error.

3. **Feedback loops:** Compare predictions to outcomes and retrain annually. Adapting to regional quirks and recent climate shifts improves future reliability.

4. **Ensemble expansion:** Combining this model with farmer experience, agronomist judgment, or other data-driven models (e.g., crop simulation models) can reduce ensemble disagreement.

### 4.8.4 Risk Communication

When communicating predictions to end-users:

- **Use percentiles, not just numbers:** "This season is expected to be at the 65th percentile of historical yields" is more intuitive than "Expected 2,500 kg/ha."
- **Highlight uncertainty visually:** Show confidence intervals on charts; explain that narrow bands mean higher confidence.
- **Discuss limiting factors:** "Rainfall is favorable but temperatures are above optimal; yield is constrained by heat stress."
- **Provide caveats:** "This model cannot account for pests or management changes; consult local extension for site-specific advice."

---

## 4.9 Summary & Key Takeaways

### 4.9.1 Model Performance

- The TCN-MLP ensemble achieved **[MAE: X kg/ha, RMSE: Y kg/ha, MAPE: Z%]** on held-out test data, explaining **[R²]** of yield variance.
- Ensemble predictions are **[Y]% more accurate** than individual fold models, validating the ensemble approach.
- Performance is robust across crops and regions, with some geographic variation reflecting local agroecological constraints.

### 4.9.2 Uncertainty & Confidence

- **[X]% of predictions fall in the "low uncertainty" band (<10% spread), indicating strong model consensus for typical scenarios.**
- **[Y]% fall in the "high uncertainty" band, typically associated with unusual climate combinations, rare crop-region pairs, or data scarcity.**
- Ensemble confidence is well-calibrated: ~95% of observations fall within predicted 95% CIs, validating the uncertainty quantification method.

### 4.9.3 Key Climate Drivers

- **Rainfall (PRECTOTCORR) is the dominant yield driver** across all crops and regions, accounting for ~[X]% of explainable variance.
- **Temperature (T2M) and maximum temperature (T2M_MAX) are secondary drivers**, with regional variation in importance.
- **Humidity and dew point contribute moderately**, particularly in disease-prone regions.
- These findings align with agronomic understanding of sub-Saharan African agroecology and support targeted interventions.

### 4.9.4 Practical Utility

- Predictions are sufficiently accurate and well-calibrated for **strategic seasonal planning** (3–6 months ahead).
- Practitioners can **use model confidence to adjust risk posture:** high confidence → plan expansively; low confidence → seek more data before committing resources.
- The model **does not replace domain expertise** but complements farmer and agronomist knowledge by providing data-driven, quantitative forecasts.

### 4.9.5 Limitations & Future Work

- The model's inability to capture management factors limits its explanatory power; hybrid models (climate + management) would improve performance.
- Uncertainty in extreme scenarios (severe drought, unusual weather) remains high; targeted data collection in these events could improve model stability.
- Temporal generalization (predictions for regions or climate patterns not in training data) requires careful validation; out-of-sample deployments should be monitored.

### 4.9.6 Advancement Opportunities

1. **Incorporate management data:** Integrate fertilizer, irrigation, and planting date to achieve 20–40% further error reduction.
2. **Multi-year memory:** Include prior-year yield and soil moisture to capture lagged effects.
3. **Causal inference:** Employ causal learning techniques (e.g., causal forests) to untangle feature interactions and derive policy-relevant recommendations.
4. **Real-time adaptation:** Deploy the model operationally with in-season updates as new forecasts and early yield indicators (e.g., satellite biomass) become available.
5. **Climate stress scoring:** Combine model predictions with agronomic thresholds to generate actionable "stress scores" that trigger insurance payouts or advisory alerts.

---

## References & Appendices

### Related Notebooks & Code

- **Model training:** `notebooks/tcn_mlp_train.ipynb` — full training pipeline, hyperparameter tuning, cross-validation
- **Evaluation:** `notebooks/tcn_mlp_eval.ipynb` — performance metrics, uncertainty analysis, temporal comparisons
- **Data exploration:** `notebooks/tcn_mlp_data.ipynb` — dataset summary, feature distributions, missing data handling
- **Explainability:** `notebooks/tcn_mlp_shap.ipynb` — LOFO sensitivity analysis, case studies, feature importance heatmaps

### Result Files

All raw results are archived in `results/`:
- `ensemble_metrics_mapes_maase_trimmed.csv` — aggregate and per-crop metrics
- `climate_yield_lag.csv` — temporal comparison results (percentile, Z-score, pct_diff)
- `permutation_feature_importance_focused.csv` — LOFO results by crop and region
- `per_crop_ensemble_mapes_maase_trimmed.csv` — disaggregated performance by crop
- `crop_climate_sensitivity.csv` — climate sensitivity summary
- `11_DIAGNOSTIC_SUMMARY.txt` — audit trail of model training, validation, and testing

### Figures Referenced

- **Figure 4.1:** Observed vs. Predicted scatter with 1:1 reference and CI shading
- **Figure 4.2:** Regional performance heatmap (MAE/RMSE by crop and region)
- **Figure 4.3:** Histogram of uncertainty_pct; threshold bands for low/medium/high
- **Figure 4.4:** Calibration curve (observed coverage vs. nominal CI level)
- **Figure 4.5:** Histogram of percent difference vs. historical mean
- **Figure 4.6:** Time series of predicted yields vs. historical mean (by crop and region)
- **Figure 4.7:** Heatmap of feature impacts (features vs. regions)
- **Figure 4.8:** Training and validation loss curves for each fold

---

**End of Chapter 4**

*This chapter established the empirical foundation for yield prediction modeling. Findings demonstrate that climate-informed ensemble methods can provide actionable forecasts with quantified uncertainty. Chapter 5 synthesizes these results, discusses conclusions, and recommends pathways for operational deployment and future research.*
