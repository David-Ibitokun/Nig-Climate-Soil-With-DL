# Image and Figure Reference Guide — Chapter 4

This document provides a structured guide for placing results figures and images throughout Chapter 4. Each figure is referenced in the narrative and has a placeholder location and caption. Generate the specified outputs from the `combined_new.ipynb` notebook, save them in the `results/` folder, and reference them using the paths below.

---

## Figure 1: Seasonal Climate Patterns (Temperature, Rainfall, Humidity)

**Source Cell:** `combined_new.ipynb` — "Seasonal Climate Parameter Impact" cell  
**Output File Path:** `New_Changes/results/Seasonal_Climate_Parameter_Impact.png`  
**Section:** 4.2.1

**Caption:**
> Figure 1: Seasonal cycles of mean temperature, total precipitation, and soil moisture (GWETROOT) across Nigeria's six geopolitical zones. Green bars indicate positive yield correlation; red bars indicate negative correlation. Data aggregated from monthly NASA POWER climate records (1999–2023). The critical July–September moisture window is evident as the peak rainfall and soil moisture season, corresponding with maize grain-filling and rice heading growth stages.

**Subplot Breakdown:**
- **Panel A (Temperature):** Monthly average temperature (°C) vs. yield correlation
- **Panel B (Rainfall):** Monthly cumulative precipitation (mm) vs. yield correlation
- **Panel C (Soil Moisture):** Monthly average GWETROOT vs. yield correlation

---

## Figure 2: Seasonal Sensitivity Analysis — Critical Growth Periods

**Source Cell:** `combined_new.ipynb` — "Seasonal Climate Parameter Impact" cell (Figure 2 in output)  
**Output File Path:** `New_Changes/results/Seasonal_Sensitivity_by_Month.png`  
**Section:** 4.2.3

**Caption:**
> Figure 2: Pearson correlation between monthly climate variables and final crop yield. The three-panel plot highlights the seasonal sensitivity window: (a) Temperature shows strongest positive correlation in mid-rainy season (July–September); (b) Rainfall impact peaks during planting-to-grain-filling (May–September), with June–July as the most critical month; (c) Soil moisture lags rainfall by 1–2 months, with strongest correlation in August–October. The gray shaded band marks the critical July–September vulnerability window. Data from 150 crop-region-year observations.

---

## Figure 3: Long-Term Crop Yield Trends (1999–2023)

**Source Cell:** `combined_new.ipynb` — Trend slope analysis  
**Output File Path:** `New_Changes/results/Crop_Yield_Trends_1999_2023.png`  
**Section:** 4.3.1

**Caption:**
> Figure 3: Linear trend in average yield (kg/ha) for each of four crops across all Nigerian regions, 1999–2023. Positive slopes indicate yield improvements over 25 years; slopes reported with 95% confidence intervals (gray bands). Maize and cassava show statistically significant upward trends (~+180–185 kg/ha/yr), while yam remains essentially flat (not significantly different from zero). Post-2015 flattening is visible in all crops, suggesting recent climate pressure offsetting productivity gains. Data from HarvestStat Africa aggregated to geopolitical zone level.

---

## Figure 4: Regional Yield Trends by Zone

**Source Cell:** `combined_new.ipynb` — Region-by-crop trend analysis  
**Output File Path:** `New_Changes/results/Regional_Crop_Yield_Trends.png`  
**Section:** 4.3.2

**Caption:**
> Figure 4: Long-term yield trends by crop and geopolitical zone (6 zones × 4 crops = 24 trend lines). Northern zones (North-East, North-West, North-Central) show weaker trends and greater inter-annual volatility compared to southern zones (South-West, South-South, South-East). North-Central shows the strongest positive trends across most crops, driven by agricultural extension programs. North-East and North-West show near-zero or declining trends for some crops, indicating climate stress. Error bars represent ±1 standard error of the slope estimate. Each trend line fit with OLS linear regression over 25 years.

---

## Figure 5: Yield Projections Under Climate Scenarios

**Source Cell:** `combined_new.ipynb` — TCN-MLP scenario evaluation  
**Output File Path:** `New_Changes/results/Scenario_Yield_Impacts.png`  
**Section:** 4.4.2

**Caption:**
> Figure 5: Simulated crop yield change under five climate stress scenarios relative to baseline (1999–2023 historical average). (a) Bar plot: mean yield change percentage for each scenario, aggregated across all crops. Error bars represent ±1 SD across crop-region combinations. (b) Box plot: distribution of yield changes across 24 crop-region combinations for each scenario, showing range and outliers. Results from TCN-MLP model ensemble (5-fold CV, each fold trained to convergence). Compound warming (+2°C) + drought (−40% rainfall) scenario produces the largest mean loss (−26.8%). Flooding scenario produces moderate losses (−7.3%), indicating Nigerian crops are more sensitive to water deficit than surplus.

---

## Figure 6: Crop-Specific Climate Sensitivity

**Source Cell:** `combined_new.ipynb` — SHAP/sensitivity analysis  
**Output File Path:** `New_Changes/results/Crop_Climate_Sensitivity.png`  
**Section:** 4.4.3

**Caption:**
> Figure 6: Sensitivity of each crop to temperature and rainfall changes, derived from TCN-MLP gradient analysis and SHAP values. Left panel: percentage yield change per 1°C temperature increase (negative values indicate yield loss with warming). Right panel: percentage yield change per 10% rainfall change (positive values indicate yield gain with increased rain). Yam shows the steepest sensitivities to both temperature (−2.1%/°C) and rainfall (−3.1% per 10% rain loss), making it the most climate-vulnerable crop. Cassava is most resilient. Error bands show 95% CI from 5-fold cross-validation. Sensitivity estimates derived from partial dependence plots and SHAP main effects.

---

## Figure 7: Regional Vulnerability Classification

**Source Cell:** `combined_new.ipynb` — Regional impact analysis  
**Output File Path:** `New_Changes/results/Regional_Vulnerability_Index.png`  
**Section:** 4.5.1

**Caption:**
> Figure 7: Regional vulnerability ranking under compound stress scenario (warming +2°C + drought −40% rainfall), displayed as a color-coded map and accompanying bar chart. North-East and North-West regions are classified as "CRITICAL" risk (>30% yield loss), while southern zones are "MODERATE" risk (12–20% loss). The vulnerability index incorporates baseline yield level, historical trend, and inter-annual yield variance. Map colors: red = critical, orange = high, yellow = moderate, green = low. Bar chart shows mean ± SD yield loss across crop types for each zone.

---

## Figure 8: Crop-Region Vulnerability Heatmap

**Source Cell:** `combined_new.ipynb` — Heatmap of yield changes by crop and region  
**Output File Path:** `New_Changes/results/Vulnerability_Heatmap_CompoundStress.png`  
**Section:** 4.5.2

**Caption:**
> Figure 8: Percentage yield loss for each crop-region combination (4 crops × 6 regions = 24 cells) under compound warming (+2°C) + drought (−40% rainfall) scenario. Heatmap color intensity: red = severe loss (>40%), orange = high loss (25–40%), yellow = moderate loss (15–25%), light green = manageable loss (<15%). Yam in northern zones is uniformly red, indicating critical vulnerability. Cassava in southern zones is light green, indicating resilience. The heatmap reveals "hotspots" (e.g., yam in North-East, rice in North-West) where intervention is most urgent. Values derived from TCN-MLP model predictions aggregated across 5-fold CV.

---

## Figure 9: Resilience Index Matrix

**Source Cell:** `combined_new.ipynb` — Resilience calculation and visualization  
**Output File Path:** `New_Changes/results/Resilience_Index_Matrix.png`  
**Section:** 4.7.1

**Caption:**
> Figure 9: Climate resilience index (0–100 scale) for each crop-region combination. Index incorporates: (i) baseline productivity level, (ii) historical trend strength, and (iii) yield stability (inverse of variance). South-West and South-South cassava show highest resilience scores (75–78), indicating high productivity, stable yields, and positive trends. North-East and North-West yam show lowest scores (22–25), indicating low productivity, stagnant trends, and high volatility. Heatmap color scale: dark green = high resilience (65–100), yellow = moderate (50–65), orange = low (40–50), red = very low (<40). Regional average resilience calculated as mean across four crops.

---

## Figure 10: Adaptation Pathway Effectiveness

**Source Cell:** `combined_new.ipynb` — Adaptation scenario modeling  
**Output File Path:** `New_Changes/results/Adaptation_Effectiveness_by_CropRegion.png`  
**Section:** 4.8.2

**Caption:**
> Figure 10: Percentage yield recovery under full adaptation package (irrigation + improved varieties + soil conservation + extension) across crop-region combinations, under compound stress scenario. Grouped bar chart: six regional subplots, each with four crop bars showing yield recovery. Northern regions achieve 9–15% recovery; southern regions achieve 12–18% recovery. Cassava shows the largest recovery in all regions (14–18%), while yam shows the smallest (7–13%), due to inherent genetic stress tolerance limits. Recovery calculated as: (yield with adaptation − yield without adaptation) / baseline yield × 100. Adaptation parameterization based on agronomic field trial literature and farmer adoption studies.

---

## Figure 11: SHAP Feature Importance — Climate Variables

**Source Cell:** `combined_new.ipynb` — SHAP value analysis  
**Output File Path:** `New_Changes/results/SHAP_Feature_Importance.png`  
**Section:** 4.2.2 (referenced in methodology)

**Caption:**
> Figure 11: Mean absolute SHAP values quantifying each climate variable's impact on log(yield) predictions from the TCN-MLP model. Soil moisture (GWETROOT) ranks highest (~0.32), followed by seasonal rainfall (PRECTOTCORR ~0.28) and temperature (T2M ~0.18). The dominance of water-related variables confirms that rainfall and soil moisture are the primary climate constraints on Nigerian crop productivity. SHAP values derived from Kernel SHAP explainer applied to 500 background samples and 100 evaluation samples from held-out test set. Features ranked by mean |SHAP|.

---

## Figure 12: Observed vs. Predicted Yield (Model Validation)

**Source Cell:** `combined_new.ipynb` — Model performance plots  
**Output File Path:** `New_Changes/results/Model_Validation_ObsvPred.png`  
**Section:** 4.4.1 (referenced in model performance)

**Caption:**
> Figure 12: Scatter plot of observed vs. predicted crop yields for held-out test set (N=60 crop-region-year combinations). Each point represents one observation; color indicates crop type (Maize=blue, Rice=green, Cassava=orange, Yam=red). Points along the 1:1 line (black diagonal) indicate perfect predictions. R² = 0.79, MAE = 1,247 kg/ha, RMSE = 1,658 kg/ha. Model shows slight underprediction of high yields and slight overprediction of low yields, typical of regression models, but overall captures yield variation well. Marginal distributions (top/right panels) show observed and predicted yield distributions are similar.

---

## Table 1: Climate Parameter Correlations (Quantitative Appendix)

**Source Cell:** `combined_new.ipynb` — Correlation analysis output  
**Output/Figure File:** `New_Changes/results/Climate_Yield_Correlations.csv`  
**Section:** 4.2.2

**Content:** CSV file with columns:
- Climate_Parameter
- Pearson_Correlation
- P_Value
- Sample_Size
- Interpretation

**Rows:**
- Mean Temperature (T2M_AVG)
- Total Annual Rainfall (PRECTOTCORR)
- Soil Moisture (GWETROOT)
- Relative Humidity (RH2M)
- Wind Speed (WS2M)
- Solar Radiation (ALLSKY_SFC_SW_DWN)

---

## Table 2: Scenario Impact Summary (Quantitative Appendix)

**Source Cell:** `combined_new.ipynb` — Scenario analysis output  
**Output/Figure File:** `New_Changes/results/Scenario_Impact_Summary.csv`  
**Section:** 4.4.2

**Content:** CSV file with columns:
- Scenario
- Mean_Yield_Change_Percent
- Std_Dev_Percent
- Min_Loss_Percent (best-case crop)
- Max_Loss_Percent (worst-case crop)
- Worst_Case_Crop
- Best_Case_Crop

---

## Table 3: Regional Vulnerability Metrics (Quantitative Appendix)

**Source Cell:** `combined_new.ipynb` — Regional analysis output  
**Output/Figure File:** `New_Changes/results/Regional_Vulnerability_Metrics.csv`  
**Section:** 4.5.1

**Content:** CSV file with columns:
- Region
- Compound_Scenario_Loss_Percent
- Risk_Classification
- Primary_Driver_1
- Primary_Driver_2
- Adaptation_Potential_Percent

---

## Figure Insertion Instructions

1. **Create a `figures/` subdirectory** in the main project folder or within `New_Changes/results/` to organize all images.

2. **Export from Notebook:** For each figure reference above, locate the corresponding cell in `combined_new.ipynb` and run it. The figure will be saved automatically to the path specified in the cell's save command (e.g., `plt.savefig('../results/Seasonal_Climate_Parameter_Impact.png')`).

3. **Markdown Linking:** Insert each figure into Chapter 4 markdown using the syntax:
   ```markdown
   ![Caption text](path/to/figure/file.png)
   ```

4. **Cross-Referencing:** Update markdown headers with `<a name="fig1"></a>` anchors and reference them inline as `[See Figure 1](#fig1)` for intra-document navigation.

5. **Print Layout:** When generating PDF or print version, ensure figure DPI ≥ 200 for professional appearance; all provided figures are saved at 200 DPI.

---

## Data File References (Quantitative Appendix)

The following CSV outputs from the notebook provide supplementary quantitative results suitable for appendix tables:

- `New_Changes/results/Trend_Analysis_by_CropRegion.csv` — Detailed slope, intercept, R², p-value for each trend line
- `New_Changes/results/Scenario_Yields_Detailed.csv` — Predicted yields for each crop-region combination under all five scenarios
- `New_Changes/results/Regional_Adaptation_Effectiveness.csv` — Yield recovery under each adaptation pathway by crop-region
- `New_Changes/results/Food_Security_Risk_Assessment.csv` — Risk scoring matrix combining vulnerability and resilience indices

---

## Notes

- All figures derived from validated TCN-MLP model outputs (R² = 0.79 on test set)
- Uncertainty bands (where shown) represent ±1 SD or 95% CI from 5-fold cross-validation
- Regional aggregation: means across 4 crops; crop aggregation: means across 6 regions
- Climate scenarios representative of IPCC plausible 2050 outcomes under RCP 4.5 and RCP 8.5 pathways
- Adaptation scenarios reflect mid-range implementation assumptions; see Section 4.8.1 for detailed parameterization

