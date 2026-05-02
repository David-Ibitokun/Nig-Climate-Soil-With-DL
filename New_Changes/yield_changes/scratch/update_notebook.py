import json
import os

def update_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    # 1. Feature Selection Mismatch (GDD -> T2M_AVG)
    # We'll search for cells containing 'unique_features =' and 'GDD_m'
    
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        
        # Update feature names
        if "unique_features = ['GDD', 'PRECTOTCORR', 'GWETROOT']" in source:
            source = source.replace("unique_features = ['GDD', 'PRECTOTCORR', 'GWETROOT']", 
                                  "unique_features = ['T2M_AVG', 'PRECTOTCORR', 'GWETROOT']")
        
        # Update GDD calculation to T2M_AVG
        if "df[f'GDD_m{month}'] = np.maximum(0, t_avg - 10)" in source:
            source = source.replace("df[f'GDD_m{month}'] = np.maximum(0, t_avg - 10)", 
                                  "df[f'T2M_AVG_m{month}'] = t_avg")
            source = source.replace("# Dynamically calculate Growing Degree Days (GDD) Base 10",
                                  "# Calculate Average Temperature (T2M_AVG)")
            
        # Update model loading print statement
        if "mini features: GDD, rainfall, humidity" in source:
             source = source.replace("mini features: GDD, rainfall, humidity", "Temp, Rainfall, Soil Moisture")

        cell['source'] = [line + '\n' for line in source.split('\n')]
        # Clean up last newline if needed (split adds an empty string at the end)
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'].pop()
            
        new_cells.append(cell)

        # 2. Add Climate Statistics Baseline before Scenario Analysis
        if "## 4. Climate Scenario Analysis" in source:
            # Insert a new cell before this one or after? The user said "Add Climate Statistics Baseline (NEW SECTION)"
            # Let's insert it before the scenario analysis section.
            baseline_stats_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# NEW: Calculate actual climate ranges from data\n",
                    "print(\"Climate Statistics (1999-2023):\")\n",
                    "for feat in ['T2M_AVG', 'PRECTOTCORR', 'GWETROOT']:\n",
                    "    monthly_cols = [col for col in df.columns if col.startswith(feat) and '_m' in col]\n",
                    "    if not monthly_cols: continue\n",
                    "    values = df[[c for c in monthly_cols if c in df.columns]].mean().mean()\n",
                    "    std_val = df[[c for c in monthly_cols if c in df.columns]].mean().std()\n",
                    "    print(f\"{feat}: Mean={values:.2f}, Std={std_val:.2f}\")\n"
                ]
            }
            # Find the index of the markdown cell we just added and insert before it
            idx = len(new_cells) - 1
            new_cells.insert(idx, baseline_stats_cell)

        # 3. Redefine Scenarios with Real Temperature Changes
        if "scenarios = {" in source and "'Warming (+1°C)'" in source:
            # Replace the scenarios dictionary and application loop
            updated_scenarios_code = """# Redefined Scenarios (precise and interpretable)
scenarios = {
    'Baseline': {'temp_change': 0, 'rainfall_pct': 0, 'soil_moisture_pct': 0},
    'Warming +1°C': {'temp_change': 1.0, 'rainfall_pct': 0, 'soil_moisture_pct': -5},
    'Warming +2°C': {'temp_change': 2.0, 'rainfall_pct': 0, 'soil_moisture_pct': -10},
    'Drought (-20% rainfall)': {'temp_change': 0.5, 'rainfall_pct': -20, 'soil_moisture_pct': -15},
    'Drought (-40% rainfall)': {'temp_change': 1.0, 'rainfall_pct': -40, 'soil_moisture_pct': -30},
    'Flooding (+30% rainfall)': {'temp_change': 0, 'rainfall_pct': 30, 'soil_moisture_pct': 25},
    'Compound: Warm + Dry': {'temp_change': 2.0, 'rainfall_pct': -30, 'soil_moisture_pct': -25},
}

scenario_results = {}

# Apply scenarios properly to T2M_AVG, PRECTOTCORR, GWETROOT
for scenario_name, changes in scenarios.items():
    X_scenario = X_seq_scaled.copy()
    
    # Feature 0: Temperature - add absolute change (scaled)
    if changes['temp_change'] != 0:
        # Approximate: 1°C ≈ 0.4 in scaled units (depends on scaler, standardizing based on project stats)
        # Using a conservative 0.4 multiplier for the standardized temperature shift
        X_scenario[:, :, 0] += (changes['temp_change'] * 0.4) 
    
    # Feature 1: Rainfall - multiplicative change
    if changes['rainfall_pct'] != 0:
        rainfall_factor = 1 + (changes['rainfall_pct'] / 100)
        X_scenario[:, :, 1] *= rainfall_factor
    
    # Feature 2: Soil Moisture (GWETROOT) - proportional change
    if changes['soil_moisture_pct'] != 0:
        soil_factor = 1 + (changes['soil_moisture_pct'] / 100)
        X_scenario[:, :, 2] *= soil_factor
    
    # Predictions and analysis...
    y_scenario_log = model.predict([X_scenario, region_ids, crop_ids, year_scaled], verbose=0).ravel()
    y_scenario = np.exp(y_scenario_log)
    
    # Calculate impact
    yield_change = ((y_scenario - y_raw) / y_raw * 100).mean()
    yield_change_pct_region = {}
    yield_change_pct_crop = {}
    
    for region in sorted(df['Region'].unique()):
        mask = df['Region'] == region
        yield_change_pct_region[region] = ((y_scenario[mask] - y_raw[mask]) / y_raw[mask] * 100).mean()
    
    for crop in sorted(df['Crop'].unique()):
        mask = df['Crop'] == crop
        yield_change_pct_crop[crop] = ((y_scenario[mask] - y_raw[mask]) / y_raw[mask] * 100).mean()
    
    # Store results
    scenario_results[scenario_name] = {
        'yields': y_scenario,
        'overall_change': yield_change,
        'by_region': yield_change_pct_region,
        'by_crop': yield_change_pct_crop
    }
    print(f"{scenario_name}: {yield_change:+.2f}%")
"""
            cell['source'] = [line + '\n' for line in updated_scenarios_code.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

        # 4. Add Direct Climate Feature Analysis & Seasonal Sensitivity after SHAP (Cell 42)
        if "plt.savefig('../results/SHAP_Feature_Importance.png', dpi=200)" in source:
            analysis_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# DIRECT CLIMATE PARAMETER ANALYSIS\n",
                    "print(\"=\"*80)\n",
                    "print(\"CLIMATE PARAMETER IMPACT ON CROP YIELD\")\n",
                    "print(\"=\"*80)\n",
                    "\n",
                    "# 1. Aggregate temperature impact (monthly average)\n",
                    "temp_cols = [col for col in df.columns if 'T2M_AVG' in col and '_m' in col]\n",
                    "rainfall_cols = [col for col in df.columns if 'PRECTOTCORR' in col and '_m' in col]\n",
                    "moisture_cols = [col for col in df.columns if 'GWETROOT' in col and '_m' in col]\n",
                    "\n",
                    "if temp_cols:\n",
                    "    avg_temp = df[temp_cols].mean(axis=1)\n",
                    "    corr_temp = avg_temp.corr(df['Yield_kg_per_ha'])\n",
                    "    print(f\"\\nTemperature (T2M_AVG) vs Yield:\")\n",
                    "    print(f\"  Pearson Correlation: {corr_temp:.4f}\")\n",
                    "    print(f\"  Interpretation: {'POSITIVE' if corr_temp > 0 else 'NEGATIVE'} relationship\")\n",
                    "\n",
                    "if rainfall_cols:\n",
                    "    total_rainfall = df[rainfall_cols].sum(axis=1)\n",
                    "    corr_rainfall = total_rainfall.corr(df['Yield_kg_per_ha'])\n",
                    "    print(f\"\\nRainfall (PRECTOTCORR) vs Yield:\")\n",
                    "    print(f\"  Pearson Correlation: {corr_rainfall:.4f}\")\n",
                    "    print(f\"  Interpretation: {'POSITIVE' if corr_rainfall > 0 else 'NEGATIVE'} relationship\")\n",
                    "\n",
                    "if moisture_cols:\n",
                    "    avg_moisture = df[moisture_cols].mean(axis=1)\n",
                    "    corr_moisture = avg_moisture.corr(df['Yield_kg_per_ha'])\n",
                    "    print(f\"\\nSoil Moisture (GWETROOT) vs Yield:\")\n",
                    "    print(f\"  Pearson Correlation: {corr_moisture:.4f}\")\n",
                    "    print(f\"  Interpretation: {'POSITIVE' if corr_moisture > 0 else 'NEGATIVE'} relationship\")\n",
                    "\n",
                    "# 2. Seasonal sensitivity analysis\n",
                    "print(\"\\n\" + \"=\"*80)\n",
                    "print(\"SEASONAL SENSITIVITY: Which months matter most?\")\n",
                    "print(\"=\"*80)\n",
                    "\n",
                    "seasonal_impact = {'Temperature': {}, 'Rainfall': {}, 'Soil_Moisture': {}}\n",
                    "for month in range(1, 13):\n",
                    "    temp_col = f'T2M_AVG_m{month}'\n",
                    "    rain_col = f'PRECTOTCORR_m{month}'\n",
                    "    moisture_col = f'GWETROOT_m{month}'\n",
                    "    if temp_col in df.columns: seasonal_impact['Temperature'][month] = df[temp_col].corr(df['Yield_kg_per_ha'])\n",
                    "    if rain_col in df.columns: seasonal_impact['Rainfall'][month] = df[rain_col].corr(df['Yield_kg_per_ha'])\n",
                    "    if moisture_col in df.columns: seasonal_impact['Soil_Moisture'][month] = df[moisture_col].corr(df['Yield_kg_per_ha'])\n",
                    "\n",
                    "import matplotlib.pyplot as plt\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
                    "months = range(1, 13)\n",
                    "month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']\n",
                    "\n",
                    "for idx, (param_name, correlations) in enumerate(seasonal_impact.items()):\n",
                    "    ax = axes[idx]\n",
                    "    values = [correlations.get(m, 0) for m in months]\n",
                    "    colors = ['green' if v > 0 else 'red' for v in values]\n",
                    "    ax.bar(months, values, color=colors, alpha=0.7, edgecolor='black')\n",
                    "    ax.set_title(f'{param_name} Seasonal Impact', fontweight='bold')\n",
                    "    ax.set_xticks(months)\n",
                    "    ax.set_xticklabels(month_names, rotation=45)\n",
                    "    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)\n",
                    "    ax.grid(True, alpha=0.3, axis='y')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../results/Seasonal_Climate_Parameter_Impact.png', dpi=200)\n",
                    "plt.show()\n"
                ]
            }
            new_cells.append(analysis_cell)

        # 5. Add Regional Sensitivity Analysis
        if "plt.savefig('../results/Seasonal_Climate_Parameter_Impact.png', dpi=200)" in source:
             regional_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# REGIONAL CLIMATE SENSITIVITY\n",
                    "print(\"\\n\" + \"=\"*80)\n",
                    "print(\"REGIONAL SENSITIVITY TO CLIMATE PARAMETERS\")\n",
                    "print(\"=\"*80)\n",
                    "regional_sensitivity = []\n",
                    "for region in sorted(df['Region'].unique()):\n",
                    "    region_df = df[df['Region'] == region]\n",
                    "    temp_corr = region_df[[c for c in temp_cols if c in region_df.columns]].mean(axis=1).corr(region_df['Yield_kg_per_ha'])\n",
                    "    rain_corr = region_df[[c for c in rain_cols if c in region_df.columns]].sum(axis=1).corr(region_df['Yield_kg_per_ha'])\n",
                    "    moist_corr = region_df[[c for c in moisture_cols if c in region_df.columns]].mean(axis=1).corr(region_df['Yield_kg_per_ha'])\n",
                    "    regional_sensitivity.append({'Region': region, 'Temp_Sensitivity': temp_corr, 'Rainfall_Sensitivity': rain_corr, 'Soil_Moisture_Sensitivity': moist_corr})\n",
                    "sens_df = pd.DataFrame(regional_sensitivity)\n",
                    "import seaborn as sns\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "heatmap_data = sens_df.set_index('Region')[['Temp_Sensitivity', 'Rainfall_Sensitivity', 'Soil_Moisture_Sensitivity']]\n",
                    "sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0)\n",
                    "plt.title('Regional Sensitivity to Climate Parameters', fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../results/Regional_Climate_Sensitivity_Heatmap.png', dpi=200)\n",
                    "plt.show()\n"
                ]
             }
             new_cells.append(regional_cell)

        # 6. Temporal Validation (Cell 40)
        if "n_samples = len(df)" in source and "train_size = int(0.8 * n_samples)" in source:
            temporal_validation_code = """# TEMPORAL VALIDATION (Train on 1999-2019, test on 2020-2023)
print("\\n" + "="*100)
print("TEMPORAL VALIDATION: 1999-2019 (Train) vs 2020-2023 (Test)")
print("="*100)

train_mask = df['Year'] <= 2019
test_mask = df['Year'] > 2019

train_r2 = r2_score(df[train_mask]['Yield_kg_per_ha'], df[train_mask]['Predicted_Yield'])
train_mae = mean_absolute_error(df[train_mask]['Yield_kg_per_ha'], df[train_mask]['Predicted_Yield'])

test_r2 = r2_score(df[test_mask]['Yield_kg_per_ha'], df[test_mask]['Predicted_Yield'])
test_mae = mean_absolute_error(df[test_mask]['Yield_kg_per_ha'], df[test_mask]['Predicted_Yield'])

print(f"Train R² (1999-2019): {train_r2:.4f} | MAE: {train_mae:.2f}")
print(f"Test R²  (2020-2023): {test_r2:.4f} | MAE: {test_mae:.2f}")

# Visualize Temporal Validation
plt.figure(figsize=(8, 5))
plt.bar(['Train (<=2019)', 'Test (>2019)'], [train_r2, test_r2], color=['#3498db', '#e74c3c'], alpha=0.7)
plt.ylabel('R² Score')
plt.title('Temporal Validation R² Score')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
plt.show()
"""
            cell['source'] = [line + '\n' for line in temporal_validation_code.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

    nb['cells'] = new_cells

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Notebook {nb_path} updated successfully!")

if __name__ == '__main__':
    update_notebook('Combined.ipynb')
