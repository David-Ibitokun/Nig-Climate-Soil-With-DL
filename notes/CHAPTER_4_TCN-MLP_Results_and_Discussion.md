# Chapter 4 — Results and Discussion (Comprehensive)

## 4.1 Overview
- Purpose: Present the experimental outcomes from the `TCN-MLP` ensemble for crop-yield prediction under historical and perturbed climate conditions, quantify predictive skill, interpret drivers of predictions, and assess robustness to ablations and scenario perturbations.
- Scope: This chapter consolidates numeric evaluation, fold-level diagnostics, spatial and temporal analyses, SHAP-based interpretability (global and local), ablation and sensitivity experiments, calibration/coverage diagnostics, and a targeted discussion of limitations and recommendations for stakeholders.

## 4.2 Methods summary (brief)
- Model: `TCN-MLP` ensemble combining temporal convolutional encoding (TCN) with an MLP predictor. Ensemble aggregates predictions across model initializations/folds to estimate mean predictions and prediction intervals.
- Data: Historical yield observations aligned with monthly climate covariates, phenological windows and management covariates (see `data/processed_dataset.csv`).
- Evaluation: 5-fold cross-validation; reported metrics are `R^2`, `MAE`, `RMSE`, `MAPE`, `sMAPE`, `MASE`. Uncertainty reported as ensemble standard deviation and 95% approximate CI widths from ensemble spread.
- Interpretability: SHAP values computed per-fold and aggregated to produce global rankings, seasonal importance, and regional summaries. Local explanations and force/waterfall plots used for case diagnostics.

## 4.3 Quantitative results
### 4.3.1 Overall summary (mean ± std across folds)
Source: `results/tcn_mlp_eval_overall_summary.csv`

| Metric | Mean | Std |
|---|---:|---:|
| Primary_R2 | 0.8255 | 0.0848 |
| Primary_MAE | 995.33 | 166.69 |
| Primary_RMSE | 2134.99 | 760.49 |
| Primary_MAPE (%) | 15.81 | 1.46 |
| Ensemble_R2 | 0.8302 | 0.0884 |
| Ensemble_MAE | 1041.07 | 176.69 |
| Ensemble_RMSE | 2105.15 | 763.46 |
| Ensemble_MAPE (%) | 16.69 | 1.31 |
| CI_Width_Mean | 1831.32 | 173.84 |

### 4.3.2 Fold-level diagnostics
Source: `results/tcn_mlp_eval_fold_metrics.csv`

| Fold | Test_Samples | Primary_R2 | Primary_MAE | Primary_RMSE | Primary_MAPE (%) | Ensemble_R2 | Ensemble_MAE | Ensemble_RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 0.9055 | 864.57 | 1423.23 | 15.21 | 0.9041 | 892.98 | 1433.11 |
| 2 | 120 | 0.7243 | 1129.68 | 3031.87 | 15.54 | 0.6984 | 1277.95 | 3171.09 |
| 3 | 120 | 0.7825 | 1103.54 | 2487.58 | 16.33 | 0.8419 | 1034.98 | 2120.91 |
| 4 | 120 | 0.7938 | 1109.77 | 2471.72 | 17.95 | 0.7933 | 1147.00 | 2474.54 |
| 5 | 120 | 0.9217 | 769.09 | 1260.53 | 13.99 | 0.9134 | 852.45 | 1326.11 |

Observations:
- Best performance observed in Fold 5 (highest `R^2`, lowest RMSE/MAE) and worst in Fold 2 (largest RMSE and lower `R^2`), indicating heterogeneous generalization likely due to regional/temporal composition differences across folds.
- Residual analysis shows heteroskedasticity: errors concentrate in low-yield quantiles and during years with strong climate anomalies (see `results/error_by_yield_quantile.csv`).

### 4.3.3 Model comparison and significance
Foldwise paired tests are recorded in `results/tcn_mlp_eval_model_comparison.csv`. Example: Fold 2 shows a significant improvement (p ≈ 0.011) when comparing the ensemble to the baseline model; other folds show mixed significance, consistent with fold heterogeneity.

## 4.4 Spatio-temporal and scenario analyses
- Temporal: Year-wise timelines (see `results/07_temporal_sensitivity_timeline.png`) indicate consistent skill across years with detectable degradations during extreme-anomaly years. Training convergence plots per fold are in `results/training_convergence_fold_*.png`.
- Spatial: Regional performance and resilience indices are available in `results/region_ensemble_metrics.csv` and `results/crop_region_resilience.csv`. Regions with lower training coverage correspond with higher ensemble predictive variance (`results/regional_mean_uncertainty_top12.png`).
- Scenario projections: Representative scenario outputs are in `results/05_climate_scenario_projections.csv` and summary impact maps in `results/06_climate_scenario_impacts.png`. Use ensemble CI widths to quantify uncertainty of scenario deltas.

## 4.5 Model interpretability — SHAP (detailed)
### 4.5.1 Global feature importance
- Global SHAP rankings (`results/shap_feature_importance.csv` and `results/SHAP_Feature_Importance.png`) consistently rank seasonal precipitation metrics, relative humidity, temperature degree-days and historical yield baseline among the top predictors.

### 4.5.2 Seasonal and regional critical windows
- Aggregated SHAP by crop and region (`results/shap_comprehensive_summary_all_crops_regions.csv`) identifies recurring critical months (Feb–Mar for many crops/regions). Example: For Maize in North-Central, `SHAP_1` (Mar) = 138.97, `SHAP_2` (Apr) = 68.27.

### 4.5.3 Interactions and dependence
- Dependence and interaction plots indicate nonlinear amplifying effects: high temperatures amplify negative impacts of low rainfall (see `results/shap_rain_vs_temp_heatmaps.png` and `results/SHAP_Precip_Dependence.png`).

### 4.5.4 Local explanations and robustness
- SHAP waterfall/force examples are provided (`results/SHAP_Waterfall_Climate_Sample.png`). To assess robustness, examine SHAP variance across ensemble members: features with high attribution variance signal uncertainty in causal importance.

## 4.6 Ablation and sensitivity experiments (numbers & implications)
### 4.6.1 Phenological ablation
Source: `results/ablation_phenological_windows.csv`.
- Summary: Ablating phenological windows produced very small deltas in aggregated MAE/R2 for the sampled experiments (example: Maize `Flowering` stage ablation changed MAE by ≈ -0.045 and R2 by ≈ -0.0014 for the sampled subset). While deltas appear numerically small in the sampled rows provided, targeted ablations in data-sparse regions and for specific crops can produce larger impacts (investigate per-crop).

### 4.6.2 Feature-subset and method comparisons
- Multi-method comparisons (`results/multi_method_comparison.csv` and `results/multi_method_comparison_rainfall.png`) show combined-weather+management inputs outperform weather-only models in MAE and RMSE.

### 4.6.3 Sensitivity to climate perturbations
Source: `results/crop_climate_sensitivity.csv` and `results/05_seasonal_sensitivity_summary.png`.
- Example patterns (Cassava): relative humidity (`RH2M`) shows strong positive correlations across most months (e.g., Jan: 0.45, Feb: 0.51), while temperature (`T2M`) displays strong negative correlations in the main growing months (Mar–Oct, correlations ~ -0.47 to -0.54). These directional sensitivities align with agronomic expectations that heat stress reduces yields while humidity/precipitation supports crop growth.

## 4.7 Calibration, coverage and reliability
- Coverage: `results/tcn_mlp_eval_coverage_summary.csv` reports full coverage over the evaluated rows (600 rows), indicating ensemble predictions produce intervals for all evaluation points.
- Calibration: Binned calibration results (`results/tcn_mlp_eval_calibration.csv`) show mean absolute gaps per prediction bin ranging from ≈ 4.7 to ≈ 1542.8 units in the largest-prediction bin; larger absolute gaps occur in the highest-prediction bins, suggesting residual under/overprediction extremes that may warrant tail-focused recalibration.
- Reliability diagram: `results/reliability_diagram.png` visualizes miscalibration; consider isotonic or quantile recalibration when deploying.

## 4.8 Discussion — limitations and implications
- Predictive vs causal: SHAP and correlation-based sensitivity highlight associations; causal inference is not established.
- Data coverage: Some regions/crops are under-represented; spatial generalization is limited where training samples are sparse (`results/spatial_generalization.csv`).
- Projection risk: Covariate shift when simulating climates outside historical ranges is a major source of uncertainty — ensemble CI widths partially capture this but do not fully account for structural model uncertainty.
- Operational recommendations: Use probabilistic thresholds, scenario stress tests, and expert review; for policy decisions, combine with domain models or causal analyses.

## 4.9 Key figures and tables for thesis
- Numeric tables: `results/tcn_mlp_eval_overall_summary.csv`, `results/tcn_mlp_eval_fold_metrics.csv`, `results/thesis_summary_table.csv`.
- SHAP figures: `results/SHAP_Feature_Importance.png`, `results/SHAP_Monthly_Importance.png`, `results/SHAP_region_*.png`.
- Scenario/uncertainty maps: `results/06_climate_scenario_impacts.png`, `results/climate_resilience_index_heatmap.png`, `results/ensemble_dashboard_highres.png`.

## 4.10 Recommendations & Next Steps
- Add explicit per-region appendices showing fold allocation, sample counts, and per-region performance tables for reproducibility.
- Include a short methods appendix describing TCN architecture, hyperparameters, ensemble aggregation method, and SHAP computation details.
- Recalibrate the ensemble in deployment using holdout subsets or quantile mapping to reduce tail miscalibration.
- Complement predictive findings with causal analysis (panel models or instrumental variables) for policy guidance.

---

### Appendix: Files referenced in this chapter
- `results/tcn_mlp_eval_overall_summary.csv`
- `results/tcn_mlp_eval_fold_metrics.csv`
- `results/tcn_mlp_eval_model_comparison.csv`
- `results/shap_comprehensive_summary_all_crops_regions.csv`
- `results/ablation_phenological_windows.csv`
- `results/crop_climate_sensitivity.csv`
- `results/tcn_mlp_eval_calibration.csv`

---

Status: Draft — comprehensive content added. I can now (choose one):
- (A) Embed selected figures directly into this Markdown and add captions.
- (B) Convert this document into a LaTeX chapter using `results/thesis_summary_table.tex` for table formatting.
- (C) Generate per-region appendix tables by parsing `results/region_ensemble_metrics.csv` and `results/spatial_generalization.csv`.

Which option would you like me to perform next?