import streamlit as st


def render():
    st.title("📘 How to use — Crop Yield Prediction App")

    st.markdown(
        """
This guide explains what each page in the app does, how to use it, and how to interpret the outputs.

App pages (what you'll find and how to use them)

1. Make Prediction (pages/01_Make_Prediction.py)
   - Purpose: Build a monthly climate profile, run the TCN-MLP ensemble, and inspect predicted yields and drivers.
   - How to use: Select `Crop`, `Region`, `Year`; edit the 12-row monthly table or upload a CSV; then click **Generate Prediction**.
   - Key outputs: expected yield range (95% CI), ensemble confidence, point estimate, and the "Drivers Behind This Prediction" table and chart.
   - Tip: Open the ⓘ expander next to the drivers for a short friendly explanation of `Yield_Impact_kg_ha`.

2. Data Explorer (pages/02_Data_Explorer.py)
   - Purpose: Browse the processed dataset that the models were trained on and that supplies climatology.
   - How to use: Filter by region, crop, and year; inspect monthly values and summary stats to ensure your inputs are realistic.

3. Model Architecture (pages/03_Model_Architecture.py)
   - Purpose: High-level description of the TCN-MLP architecture and training design decisions.
   - How to use: Read this page if you want background on model structure; not required for everyday use.

4. Model Evaluation (pages/04_Model_Evaluation.py)
   - Purpose: Validation metrics and error breakdowns (MAPE, RMSE, bias) across folds, crops, and regions.
   - How to use: Check these metrics to understand where the model performs well or poorly.

5. How to Use (this page — pages/15_How_To_Use.py)
   - Purpose: Step-by-step instructions, page descriptions, tips, and troubleshooting.

6. Historical Trends (pages/06_Historical_Trends.py)
   - Purpose: Visualize long-term climate and yield trends to provide context for predictions.

7. Ensemble Analysis (pages/07_Ensemble_Analysis.py)
   - Purpose: Inspect individual ensemble members, their spread, and how they contribute to uncertainty.

8. Climate Patterns (pages/08_Climate_Patterns.py)
   - Purpose: Diagnostics of climate regimes, monthly importance summaries, and anomaly visualizations.

9. Recommendations (pages/10_Recommendations.py)
   - Purpose: High-level, actionable recommendations derived from model outputs and diagnostics.

10. About (pages/99_About.py)
   - Purpose: Project background, data sources, and contact/attribution information.

Practical guide: using predictions responsibly

- Treat the predicted range (95% CI) and ensemble confidence as the primary decision aids.
- Use `Yield_Impact_kg_ha` to rank features and build intuition about drivers — but remember the caveats in the info expander.
- If ensemble confidence is low or evaluation errors are large for your crop/region, interpret predictions cautiously.

Troubleshooting & tips

- Zero rainfall: provide non-zero `PRECTOTCORR` in the monthly profile or enable the force option (not recommended).
- Missing artifacts (scalers, label mappings): the app will fit fallbacks; results remain useful but may change slightly from production runs.
- Slow startup: the first model load may take time while TensorFlow models are cached.

Glossary (short)

- `Yield_Impact_kg_ha`: model-estimated change in predicted yield (kg/ha) when a feature is replaced by climatology — useful for ranking drivers.
- `Climatology` / `Baseline`: historical median monthly values from the processed dataset used as the reference.
- `Ensemble Confidence`: heuristic based on spread between ensemble members; higher = more agreement.

Want it linked from the main menu or a quick "Open Prediction" button here? Tell me where you want the shortcut and I'll add it.
"""
    )


if __name__ == "__main__":
    render()
