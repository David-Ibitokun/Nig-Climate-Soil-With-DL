# Actionable Improvement Checklist
## Code Snippets to Add to Your Notebooks

---

## 1. TRAIN NOTEBOOK IMPROVEMENTS

### 1.1 Add Learning Curves (Monitor Convergence)

**Add this cell after training each fold:**

```python
# Visualize learning curves for best-performing fold (if history saved)
import matplotlib.pyplot as plt

fold_num = best_fold  # or loop through folds
history = fold_histories[fold_num]  # must save model.fit() history

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training loss
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Model Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# R² over epochs (if tracked)
if 'r2' in history.history:
    axes[1].plot(history.history['r2'], label='Train R²', linewidth=2)
    axes[1].plot(history.history['val_r2'], label='Val R²', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Model Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/training_convergence_fold_{}.png'.format(fold_num), dpi=300, bbox_inches='tight')
plt.show()

print(f"Fold {fold_num}: Final train loss = {history.history['loss'][-1]:.6f}")
print(f"Fold {fold_num}: Final val loss = {history.history['val_loss'][-1]:.6f}")
print(f"Fold {fold_num}: Improvement = {history.history['val_loss'][0] - history.history['val_loss'][-1]:.6f}")
```

**Why**: Demonstrates model actually learned (not random). Shows convergence and any overfitting.

---

### 1.2 Add Hyperparameter Documentation

**Add markdown cell:**

```markdown
## Hyperparameter Summary

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 3e-4 | Small dataset (600), needs careful learning |
| Weight Decay | 3e-4 | Additional L2 regularization |
| L2 Regularization (kernel) | 1e-3 | Prevent overfitting |
| Dropout Rate | 0.12 | Moderate; high dropout = underfitting on small data |
| Batch Size | [your value] | Use small batch (16-32) for dataset this size |
| Epochs | [your value] | Stop when val_loss plateaus |
| Conv1D Filters | 48 → 32 | Taper to avoid parameter explosion |
| Conv1D Kernel | 3 | Captures 3-month patterns (typical agricultural cycles) |
| Dense Units | 64 → 32 | Bottleneck to reduce overfitting |
| Embedding Dim | [your value] | Crop/region embeddings; typical = 8-16 |

**Rationale**: Chosen to balance model capacity with data scarcity (600 samples).
Over-parameterized models overfit; under-parameterized models underfit.
Final ensemble uses all 5 folds to reduce variance.
```

---

### 1.3 Add Seed & Reproducibility Statement

**Add cell:**

```python
# Reproducibility Verification
print("="*80)
print("REPRODUCIBILITY CHECK")
print("="*80)
print(f"Random Seed: {SEED}")
print(f"OS PYTHONHASHSEED: {os.environ['PYTHONHASHSEED']}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print()
print("To reproduce:")
print(f"  1. Set SEED = {SEED}")
print(f"  2. Run notebook top-to-bottom in a fresh kernel")
print(f"  3. Use same library versions (see above)")
print("="*80)
```

---

## 2. EVAL NOTEBOOK IMPROVEMENTS

### 2.1 Add Temporal Trend Analysis

**Add this cell in the climate sensitivity section:**

```python
# CRITICAL: Show that climate variables are actually *changing* over time
# This validates the climate change thesis

import scipy.stats as stats

print("="*80)
print("CLIMATE TRENDS (1999-2023): Evidence of Climate Change")
print("="*80)

climate_trends = []

for region in sorted(df['Region'].unique()):
    region_data = df[df['Region'] == region].groupby('Year').agg({
        'PRECTOTCORR_m5': 'mean',  # May rainfall
        'PRECTOTCORR_m6': 'mean',  # June rainfall
        'T2M_m7': 'mean',           # July temperature
        'T2M_m8': 'mean',           # August temperature
    }).reset_index()
    
    for feature in ['PRECTOTCORR_m5', 'T2M_m7']:
        if feature in region_data.columns:
            x = region_data['Year'].values
            y = region_data[feature].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine if trend is significant
            significant = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
            direction = "↑ INCREASING" if slope > 0 else "↓ DECREASING"
            
            climate_trends.append({
                'Region': region,
                'Feature': feature,
                'Trend/Year': slope,
                'R²': r_value**2,
                'p_value': p_value,
                'Significant': significant,
                'Direction': direction
            })
            
            print(f"\n{region} - {feature}:")
            print(f"  Slope: {slope:.4f} units/year {direction} {significant}")
            print(f"  p-value: {p_value:.4f} {'(SIGNIFICANT)' if p_value < 0.05 else '(not significant)'}")
            print(f"  25-year change: {slope * 24:.2f} units (from 1999 to 2023)")

trends_df = pd.DataFrame(climate_trends)
trends_df.to_csv('../results/climate_trends_1999_2023.csv', index=False)

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("Positive slope = Warming (temperature) or Increasing (rainfall)")
print("Negative slope = Cooling (temperature) or Decreasing (rainfall)")
print("p < 0.05 = Trend is statistically significant")
print("p ≥ 0.05 = Trend not significant (could be noise)")
print()
print("For thesis: Show that climate variables SHAP identifies as important")
print("           are actually CHANGING over 1999-2023.")
print("           This tightens the causal argument:")
print("           Climate changes → Model says climate drives yield → Food security threatened")
```

**Visualization to add:**

```python
# Visualize climate trends
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
regions = sorted(df['Region'].unique())

for idx, region in enumerate(regions):
    ax_row = idx // 3
    ax_col = idx % 3
    ax = axes[ax_row, ax_col]
    
    region_data = df[df['Region'] == region].groupby('Year').agg({
        'PRECTOTCORR_m5': 'mean',  # Rainfall (May)
        'T2M_m7': 'mean'            # Temperature (July)
    }).reset_index()
    
    # Normalize to same scale for comparison
    rainfall_norm = (region_data['PRECTOTCORR_m5'] - region_data['PRECTOTCORR_m5'].min()) / \
                    (region_data['PRECTOTCORR_m5'].max() - region_data['PRECTOTCORR_m5'].min())
    temp_norm = (region_data['T2M_m7'] - region_data['T2M_m7'].min()) / \
                (region_data['T2M_m7'].max() - region_data['T2M_m7'].min())
    
    ax.plot(region_data['Year'], rainfall_norm, 'o-', label='May Rainfall (norm)', linewidth=2, markersize=4)
    ax.plot(region_data['Year'], temp_norm, 's-', label='July Temp (norm)', linewidth=2, markersize=4)
    
    # Add trend lines
    z_rain = np.polyfit(region_data['Year'], rainfall_norm, 1)
    p_rain = np.poly1d(z_rain)
    z_temp = np.polyfit(region_data['Year'], temp_norm, 1)
    p_temp = np.poly1d(z_temp)
    
    ax.plot(region_data['Year'], p_rain(region_data['Year']), '--', alpha=0.7, color='blue')
    ax.plot(region_data['Year'], p_temp(region_data['Year']), '--', alpha=0.7, color='orange')
    
    ax.set_title(f'{region}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Climate Trends by Region (1999-2023)\nThicker trend line = stronger change', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('../results/climate_trends_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Why**: Closes the loop on climate change thesis. "SHAP says rainfall matters. Data shows rainfall is decreasing in North. Therefore climate change threatens yields."

---

### 2.2 Add Temporal Alignment Documentation

**Add markdown cell at start:**

```markdown
## DATA TEMPORAL ALIGNMENT (CRITICAL)

**Question**: For planting year Y, which months of climate do we use?

**Answer** (document your actual design):

Option A: Calendar Year Alignment
- Year 2020 → Climate Jan 2020 - Dec 2020
- Assumption: Planting cycle exactly matches calendar year
- Valid if: Crops planted in January and harvested in December
- Risk: Does this match Nigerian agricultural calendar?

Option B: Agricultural Year Alignment
- Planting Year 2020 → Climate Jan 2019 - Dec 2020 (covers preceding dry season)
- Assumption: Planting in May 2020 preceded by prior-year conditions
- Valid if: Farmers make decisions based on previous year's climate
- Risk: Includes data from 2019 (temporal leakage if 2019 data is "future" for model training)

Option C: Backward-Looking Window
- Year 2020 → Climate Jul 2019 - Jun 2020 (12-month window preceding harvest)
- Assumption: Matches typical West African monsoon (Jun-Sep rain, Nov-Feb dry)
- Valid if: This captures the actual growing season
- Risk: Must verify against FAO crop calendars for Nigeria

**This Project Uses**: [State which option; e.g., "Option A - Calendar Year"]

**Justification**: 
- HarvestStat labels yields as "planting_year"
- No explicit planting dates in data
- Calendar year alignment is conservative (avoids temporal assumptions)
- [Add any additional reasoning]

**Validation**:
- Sanity check: Do May-August months have higher rainfall? (Yes/No)
- Correlation check: Does lag-0 (same-year) rainfall correlate with yield more than lag-1? (Yes/No)
- Literature check: Do findings match FAO crop calendars? (Yes/No)
```

---

### 2.3 Add Skill Score Calculation

**Add in metrics section:**

```python
# Skill Score: Compare model to naive baseline
# Naive baseline = always predict crop's historical mean yield

def compute_skill_score(y_true, y_pred, y_baseline):
    """
    Skill Score = 1 - (MSE_model / MSE_baseline)
    Interpretation:
      SS = 1.0: Model perfect
      SS = 0.0: Model = baseline (useless)
      SS < 0.0: Model worse than baseline (bad!)
    """
    mse_model = np.mean((y_true - y_pred)**2)
    mse_baseline = np.mean((y_true - y_baseline)**2)
    ss = 1 - (mse_model / mse_baseline)
    return ss

# For each crop, compute baseline = mean yield of that crop
crop_baseline = {}
for crop in df['Crop'].unique():
    crop_baseline[crop] = df[df['Crop']==crop]['Yield_kg_per_ha'].mean()

# Add baseline column
eval_df['Baseline_Yield'] = eval_df['Crop'].map(crop_baseline)

# Compute skill scores
eval_df['Skill_Score'] = eval_df.apply(
    lambda row: compute_skill_score(
        row['Yield_True'], 
        row['Yield_Ensemble'], 
        row['Baseline_Yield']
    ), 
    axis=1
)

print("\nSkill Score Summary:")
print(f"  Mean: {eval_df['Skill_Score'].mean():.3f}")
print(f"  Std:  {eval_df['Skill_Score'].std():.3f}")
print(f"  Min:  {eval_df['Skill_Score'].min():.3f}")
print(f"  Max:  {eval_df['Skill_Score'].max():.3f}")

print("\nInterpretation:")
if eval_df['Skill_Score'].mean() > 0.7:
    print("  ✓ Excellent: Model much better than naive baseline")
elif eval_df['Skill_Score'].mean() > 0.5:
    print("  ✓ Good: Model meaningfully better than baseline")
elif eval_df['Skill_Score'].mean() > 0.3:
    print("  ⚠ Fair: Model better than baseline, but limited skill")
else:
    print("  ✗ Poor: Model barely better than baseline")
```

---

## 3. SHAP NOTEBOOK IMPROVEMENTS

### 3.1 Add SHAP Background Data Validation

**Add early in notebook after loading data:**

```python
# Validate that SHAP background data is representative
print("="*80)
print("SHAP BACKGROUND DATA VALIDATION")
print("="*80)

# Typical: use 10-20% of training data as background
background_size = min(100, len(X_seq) // 5)  # ~20% or max 100
background_idx = np.random.choice(len(X_seq), background_size, replace=False)

# Check representativeness
bg_crops = crop_ids[background_idx]
bg_regions = region_ids[background_idx]
bg_years = year_ids[background_idx]

print(f"Total samples: {len(X_seq)}")
print(f"Background samples: {background_size} ({100*background_size/len(X_seq):.1f}%)")

print("\nBackground distribution:")
print("\nCrop representation:")
for crop in np.unique(crop_ids):
    n_total = np.sum(crop_ids == crop)
    n_bg = np.sum(bg_crops == crop)
    pct = 100 * n_bg / background_size
    print(f"  {CROP_NAMES[crop]}: {n_bg}/{background_size} ({pct:.1f}%) [total: {n_total}]")

print("\nRegion representation:")
for region in np.unique(region_ids):
    n_total = np.sum(region_ids == region)
    n_bg = np.sum(bg_regions == region)
    pct = 100 * n_bg / background_size
    print(f"  {REGION_NAMES[region]}: {n_bg}/{background_size} ({pct:.1f}%) [total: {n_total}]")

print("\nYear range:")
print(f"  Background years: {bg_years.min()}-{bg_years.max()}")
print(f"  Total years: {year_ids.min()}-{year_ids.max()}")

# Check if distribution is balanced
chi2_crop = np.sum((bg_crops[None, :] == np.unique(crop_ids)[:, None]).sum(axis=1) ** 2)
chi2_region = np.sum((bg_regions[None, :] == np.unique(region_ids)[:, None]).sum(axis=1) ** 2)

print("\nBalance check:")
print(f"  Crop χ²: {chi2_crop:.2f} (lower = more balanced)")
print(f"  Region χ²: {chi2_region:.2f} (lower = more balanced)")

print("\n" + "="*80)
print("RECOMMENDATION:")
if chi2_crop < 50 and chi2_region < 50:
    print("✓ Background data is well-balanced across crops and regions")
    print("  SHAP attribution should be reliable")
else:
    print("⚠ Background data is skewed toward certain crops/regions")
    print("  Consider re-sampling for balance, or documenting the bias")
print("="*80)
```

---

### 3.2 Add Cross-Fold SHAP (for Leakage-Free Attribution)

**Optional: If you want maximum rigor, replace full-data SHAP with this:**

```python
# CROSS-FOLD SHAP: Run SHAP on each fold's test set using that fold's model
# This avoids train-test leakage and provides uncertainty on SHAP values

print("="*80)
print("COMPUTING CROSS-FOLD SHAP (Leakage-Free Attribution)")
print("="*80)

all_shap_values = []  # Collect SHAP values across folds
all_indices = []      # Track which sample came from which fold

for fold in range(1, n_folds + 1):
    print(f"\nFold {fold}/{n_folds}:")
    
    # Load fold-specific model and scaler
    fold_model = keras.models.load_model(f'../models/tcn_mlp_fold_{fold}.keras', compile=False)
    fold_scaler = scalers[fold]
    
    # Get test set for this fold
    _, test_idx = cv_splits[fold]
    X_test_fold = X_seq[test_idx]
    y_test_fold = y_raw[test_idx]
    
    # Use test set samples + some training samples as background
    # (Only fold training data, to avoid leakage)
    train_idx = cv_splits[fold][0]
    bg_idx = np.random.choice(train_idx, size=min(50, len(train_idx)), replace=False)
    
    X_bg = X_seq[bg_idx]
    
    # Create SHAP explainer using fold's background
    explainer = shap.DeepExplainer(fold_model, X_bg)
    
    # Compute SHAP on fold's test set
    shap_fold = explainer.shap_values(X_test_fold)
    
    # Average across months (aggregate over temporal dimension)
    shap_fold_aggregated = np.mean(np.abs(shap_fold), axis=1)  # (n_samples, n_features)
    
    all_shap_values.append(shap_fold_aggregated)
    all_indices.extend([(fold, idx) for idx in test_idx])
    
    print(f"  SHAP computed for {len(X_test_fold)} test samples")
    print(f"  Mean |SHAP|: {np.mean(np.abs(shap_fold)):.4f}")

# Aggregate across folds
all_shap_values_array = np.vstack(all_shap_values)
shap_mean = np.mean(all_shap_values_array, axis=0)
shap_std = np.std(all_shap_values_array, axis=0)

print("\n" + "="*80)
print("CROSS-FOLD SHAP SUMMARY")
print("="*80)
print("Global feature importance (mean |SHAP| across all folds' test sets):")
for i, feature in enumerate(CLIMATE_FEATURES):
    print(f"  {feature}: {shap_mean[i]:.4f} ± {shap_std[i]:.4f}")

print("\nNote: Standard deviation reflects uncertainty from fold-to-fold variation")
print("      If some features vary widely across folds → importance is unstable")
```

---

### 3.3 Add Causal Language Check

**Add markdown cell before interpreting SHAP:**

```markdown
## Causal Language Guidance

### DO SAY:
- "Rainfall *contributes* to yield variation"
- "Our model shows rainfall is *associated with* higher yields"
- "SHAP indicates rainfall has the *largest feature importance*"
- "Climate factors *explain* 76% of yield variance"

### DON'T SAY:
- "Rainfall *causes* yield" (could be confounding by soil, farmer behavior, etc.)
- "Climate *determines* food security" (socio-economic factors also matter)
- "Temperature *reduces* yields" (might be correlated with dry season, not causal)
- "Our model proves food security will decline" (model ≠ reality; future is uncertain)

### BETTER FRAMING:
"Our analysis reveals that, conditioned on the trained neural network's learned representation, rainfall variation is the dominant driver of predicted yield variation. While not strictly causal (no randomized experiment), this finding aligns with agronomic literature and stable patterns across multiple validation methods, suggesting that climate monitoring/forecasting could improve yield predictions."
```

---

### 3.4 Add Contingency Analysis

**Add to validate non-linearity:**

```python
# Contingency analysis: Does the effect of rainfall depend on temperature?
# This shows interactions, not just main effects

print("="*80)
print("INTERACTION ANALYSIS: Does rainfall effect depend on temperature?")
print("="*80)

# Divide samples into temperature quartiles
temp_quartiles = np.quantile(df['T2M_m7'].values, [0, 0.25, 0.5, 0.75, 1.0])

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for q in range(4):
    temp_min, temp_max = temp_quartiles[q], temp_quartiles[q+1]
    mask = (df['T2M_m7'] >= temp_min) & (df['T2M_m7'] < temp_max)
    
    rainfall_vals = df.loc[mask, 'PRECTOTCORR_m5'].values
    yield_vals = df.loc[mask, 'Yield_kg_per_ha'].values
    
    # Correlation in this temperature range
    corr = np.corrcoef(rainfall_vals, yield_vals)[0, 1]
    
    axes[q].scatter(rainfall_vals, yield_vals, alpha=0.5, s=30)
    
    # Fit line
    z = np.polyfit(rainfall_vals, yield_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rainfall_vals.min(), rainfall_vals.max(), 100)
    axes[q].plot(x_line, p(x_line), 'r--', linewidth=2)
    
    axes[q].set_title(f'Temp: {temp_min:.1f}-{temp_max:.1f}°C\ncorr(rain,yield)={corr:.2f}')
    axes[q].set_xlabel('May Rainfall (mm)')
    axes[q].set_ylabel('Yield (kg/ha)')
    axes[q].grid(True, alpha=0.3)

plt.suptitle('Rainfall-Yield Relationship at Different Temperatures\nIf slopes differ → rainfall effect is conditional on temperature (interaction)', 
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('../results/interaction_rainfall_temperature.png', dpi=300, bbox_inches='tight')
plt.show()

print("Interpretation:")
print("If rainfall-yield correlation is similar across temperature quartiles")
print("  → No interaction (rainfall effect is independent of temperature)")
print("If correlation varies across temperature ranges")
print("  → Interaction exists (effect of rain depends on temperature)")
```

---

## 4. GENERAL IMPROVEMENTS (All Notebooks)

### 4.1 Add Glossary Cell

**At end of each notebook:**

```markdown
## Glossary of Terms

| Term | Definition | Unit |
|------|-----------|------|
| T2M | Temperature at 2 meters | °C |
| PRECTOTCORR | Precipitation (corrected) | mm/month |
| RH2M | Relative Humidity at 2m | % |
| SHAP | SHapley Additive exPlanations | kg/ha (in this context) |
| R² | Coefficient of Determination | unitless, 0-1 |
| RMSE | Root Mean Squared Error | kg/ha |
| MAPE | Mean Absolute Percentage Error | % |
| Ensemble | Average of 5 fold models | predictions |
| Cross-Validation | k-fold (k=5) stratified | avoids leakage |
| Phenological Window | Critical growth stage | calendar months |
```

---

### 4.2 Add Bibliography/Hyperlinks

**At end of notebooks:**

```markdown
## References

- NASA POWER: https://power.larc.nasa.gov/ [climate data source]
- HarvestStat Africa: https://github.com/HarvestStat/HarvestStat-Africa [yield data]
- Lee et al. (2025): https://doi.org/10.1038/s41597-025-05001-z [harmonized crop statistics]
- FAO Crop Calendars: https://www.fao.org/land-water/crops/agricultural-calendars [crop phenology]
- SHAP Library: https://github.com/slundberg/shap [feature attribution]
- Explainability in Agriculture: [papers on XAI for crop models]
```

---

## 5. SUBMISSION CHECKLIST

### Before Final Submission:

- [ ] All 3 notebooks run top-to-bottom without errors
- [ ] Learning curves added to train notebook
- [ ] Climate trends visualization added to eval notebook
- [ ] SHAP background data validated
- [ ] Temporal alignment documented
- [ ] Interaction analysis included
- [ ] All figures saved to `../results/` with high DPI (300)
- [ ] All tables exported to CSV
- [ ] Code is commented (non-obvious lines)
- [ ] No hardcoded paths (all relative paths)
- [ ] Random seed explicitly set and documented
- [ ] Reproducibility section added

### Thesis Integration:

- [ ] Results figures/tables copied to thesis (with proper captions)
- [ ] SHAP findings connected to climate change argument
- [ ] Per-crop and per-region insights highlighted
- [ ] Limitations and caveats explicitly stated
- [ ] Future work section suggests next steps

---

## EXPECTED TIMELINE FOR IMPROVEMENTS

| Improvement | Time | Priority |
|------------|------|----------|
| Climate trends analysis (Sec 2.1) | 30 min | **HIGH** – closes climate change argument |
| Learning curves (Sec 1.1) | 20 min | **HIGH** – shows convergence |
| SHAP background validation (Sec 3.1) | 20 min | **HIGH** – validates SHAP rigor |
| Skill score (Sec 2.3) | 15 min | **MEDIUM** – contextualizes performance |
| Interaction analysis (Sec 3.4) | 30 min | **MEDIUM** – reveals non-linearity |
| Temporal alignment doc (Sec 2.2) | 10 min | **HIGH** – critical for thesis |
| Cross-fold SHAP (Sec 3.2) | 45 min | **LOW** – optional but rigorous |
| Causal language check (Sec 3.3) | 10 min | **MEDIUM** – avoid overstatements |

**Total time: 2-3 hours** for all improvements

**Minimum time (HIGH priority only): 1 hour**

---

**You're very close. These additions will elevate from "good" to "excellent."** ✅

Good luck with your defense! 🎓
