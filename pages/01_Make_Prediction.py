import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import traceback
import io
import zipfile
from pathlib import Path

import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from data_loader import apply_global_style, load_data
from scripts.prediction_helpers import (
    CLIMATE_FEATURES,
    build_feature_sensitivity_summary,
    build_global_climatology_sequence,
    build_year_features_arr,
    compute_climate_anomalies,
    compute_historical_yield_baseline,
    compute_regional_climate_baseline,
    get_feature_explanation,
    get_month_labels,
    load_ensemble_models,
    load_prediction_artifacts,
    predict_ensemble_yield,
)
from scripts.prediction_report import (
    build_prediction_report_filename,
    build_prediction_report_markdown,
    build_prediction_report_pdf_bytes,
)

# Small constant used in log-transform inversion
EPSILON = 1e-6

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_INVENTORY_PATH = RESULTS_DIR / "tcn_mlp_model_inventory.csv"
X_SCALER_PATH = MODEL_DIR / "x_scaler.pkl"
YEAR_SCALER_PATH = MODEL_DIR / "year_scaler.pkl"
CROP_STATS_PATH = MODEL_DIR / "crop_yield_stats.pkl"
LABEL_MAPPINGS_PATH = MODEL_DIR / "label_mappings.json"

CLIMATE_FEATURE_LABELS = {
    "T2M": "Mean temperature at 2m",
    "T2M_MAX": "Max temperature at 2m",
    "T2M_MIN": "Min temperature at 2m",
    "TS": "Land surface temperature",
    "T2MDEW": "Dew point temperature at 2m",
    "T2MWET": "Wet bulb temperature at 2m",
    "PRECTOTCORR": "Bias-corrected total precipitation",
    "RH2M": "Relative humidity at 2m",
    "QV2M": "Specific humidity at 2m",
}

CLIMATE_FEATURES = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "TS",
    "T2MDEW",
    "T2MWET",
    "PRECTOTCORR",
    "RH2M",
    "QV2M",
]

FEATURE_EXPLANATIONS = {
    "PRECTOTCORR": "Precipitation supplies water. More rain generally increases yield up to an optimal range; too little reduces yield, too much can harm crops via flooding or leaching.",
    "T2M": "Mean air temperature affects development rate; deviations from crop-optimal ranges (too hot or too cold) reduce yields via stress or slower growth.",
    "T2M_MAX": "High daytime maxima can cause heat stress, reduce grain filling and pollination success, lowering yields.",
    "T2M_MIN": "Low night temperatures can increase respiration losses or cause cold stress; extremes reduce yield.",
    "TS": "Land surface temperature reflects canopy and soil heating; extreme values can indicate stress that reduces yield.",
    "T2MDEW": "Dew point indicates air moisture; low dew points mean drier air and higher evapotranspiration, which can reduce yields under water stress.",
    "T2MWET": "Wet-bulb temperature captures combined heat and humidity; high wet-bulb increases heat stress severity under humid conditions.",
    "RH2M": "Relative humidity influences evapotranspiration and disease risk; low RH increases water loss, high RH can favor disease—both affect yield depending on context.",
    "QV2M": "Specific humidity measures absolute moisture content; low specific humidity usually signals drier air and greater water stress on plants.",
}


def _export_prediction_chart_png(y_pred: float, y_lower: float, y_upper: float, fig: go.Figure) -> bytes | None:
    try:
        return fig.to_image(format="png", engine="kaleido", scale=2)
    except Exception:
        try:
            return fig.to_image(format="png", engine="kaleido", scale=1)
        except Exception:
            try:
                return fig.to_image(format="png")
            except Exception:
                try:
                    fig_mpl, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                    ax.bar(["Yield"], [y_pred], color="#2E86AB", width=0.6)
                    ax.errorbar(
                        ["Yield"],
                        [y_pred],
                        yerr=[[y_pred - y_lower], [y_upper - y_pred]],
                        fmt="none",
                        ecolor="#A1C6D4",
                        elinewidth=2,
                        capsize=8,
                    )
                    ax.set_ylabel("Yield (kg/ha)")
                    ax.set_title("Yield Prediction with uncertainty range")
                    ax.grid(axis="y", alpha=0.25)
                    fig_mpl.tight_layout()
                    buffer = io.BytesIO()
                    fig_mpl.savefig(buffer, format="png", bbox_inches="tight")
                    plt.close(fig_mpl)
                    buffer.seek(0)
                    return buffer.getvalue()
                except Exception:
                    return None


def _export_driver_chart_png(driver_display: pd.DataFrame, chart_mode: str, x_axis_title: str) -> bytes | None:
    try:
        fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=200)
        values = pd.to_numeric(driver_display["Yield_Impact_kg_ha"], errors="coerce").fillna(0.0).astype(float)
        if chart_mode == "Absolute" and "Normalized_Abs_Impact_kg_ha" in driver_display.columns:
            abs_col = pd.to_numeric(driver_display["Normalized_Abs_Impact_kg_ha"], errors="coerce").fillna(0.0).astype(float)
            values = abs_col
            colors = ["#2E86AB"] * len(values)
        else:
            colors = ["#2E86AB" if value >= 0 else "#D1495B" for value in values]
        features = driver_display["Feature"].tolist()
        ax.barh(features, values, color=colors)
        ax.axvline(0, color="#6b7280", linewidth=1)
        ax.set_xlabel(x_axis_title)
        ax.set_title("Feature Influence on Yield")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception:
        return None


def render():
    apply_global_style()
    
    st.title("🎯 Make Prediction")
    
    st.markdown("""
Use the TCN-MLP Ensemble model to predict crop yields based on climate parameters.
Select your inputs and the model will provide yield predictions with confidence intervals.
    """)
    
    data = load_data()
    processed_dataset = data.get("processed_dataset") if isinstance(data.get("processed_dataset"), pd.DataFrame) else pd.DataFrame()

    if not processed_dataset.empty:
        crop_options = sorted(processed_dataset["Crop"].dropna().unique().tolist())
        region_options = sorted(processed_dataset["Region"].dropna().unique().tolist())
        year_min, year_max = 2026, 2030
    else:
        crop_options = ["Maize", "Rice", "Cassava", "Yam"]
        region_options = ["North-Central", "North-East", "North-West", "South-East", "South-South", "South-West"]
        year_min, year_max = 2026, 2030

    # Prediction interface
    st.subheader("📋 Select Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop = st.selectbox(
            "Select Crop",
            crop_options
        )
    
    with col2:
        region = st.selectbox(
            "Select Region",
            region_options
        )
    
    with col3:
        year = st.slider(
            "Select Year",
            min_value=year_min,
            max_value=year_max,
            value=year_max
        )
    
    st.markdown("---")
    
    # Advanced: build monthly sequences
    st.subheader("🧾 Monthly Climate Sequence")
    with st.expander("📚 Climate Parameter Reference", expanded=False):
        st.caption(
            "Parameter_Name is the human-readable description of each climate variable, while Parameter_Code is the short code used in the dataset and model inputs."
        )
        st.code(
            """Category,       Parameter_Code, Parameter_Name,                      Unit
    temperature,    T2M,            Mean temperature at 2m,             °C
    temperature,    T2M_MAX,        Max temperature at 2m,              °C
    temperature,    T2M_MIN,        Min temperature at 2m,              °C
    temperature,    TS,             Land surface temperature,           °C
    rainfall,       PRECTOTCORR,    Bias-corrected total precipitation, mm/month
    humidity,       RH2M,           Relative humidity at 2m,            %
    humidity,       QV2M,           Specific humidity at 2m,            g/kg
    humidity,       T2MDEW,         Dew point temperature at 2m,        °C
    humidity,       T2MWET,         Wet bulb temperature at 2m,         °C"""
        )
        st.markdown(
            """
**Parameter_Name explanations**

- **Mean temperature at 2m**: the average air temperature measured near the ground surface (°C).
- **Max temperature at 2m**: the hottest air temperature near the ground surface for the month (°C).
- **Min temperature at 2m**: the coldest air temperature near the ground surface for the month (°C).
- **Land surface temperature**: the temperature of the land itself, not the air above it (°C).
- **Bias-corrected total precipitation**: ⚠️ **the total monthly rainfall in mm** (NOT mm/day). Sum all daily rainfall for the month to get one value per month.
- **Relative humidity at 2m**: how much moisture is in the air near the ground, compared with the maximum possible (%).
- **Specific humidity**: the actual amount of water vapor in the air (g/kg).
- **Dew point temperature at 2m**: the temperature at which air near the ground becomes saturated and condensation starts (°C).
- **Wet bulb temperature at 2m**: the temperature reached when air is cooled by evaporation; it reflects heat and moisture together (°C).

**📌 How to provide your data:**

1. **Temperature features** (T2M, T2M_MAX, T2M_MIN, TS): Provide the monthly average or aggregated value from daily observations (°C).
2. **Precipitation (PRECTOTCORR)**: Provide the **monthly total** in mm. If you have daily data, sum all days in the month.
3. **Humidity features** (RH2M, QV2M, T2MDEW, T2MWET): Provide the monthly average (% or g/kg for QV2M; °C for dew/wet-bulb temperatures).
4. **All 12 rows**: One row per month (Jan–Dec), in order. Your sequence should represent a complete annual climate profile.

**⚠️ Important**: Do not provide daily averages for monthly fields. Use monthly aggregates. If unsure, use the Download sample CSV button and open it in Excel.
            """
        )

    seq_mode = st.radio(
        "Sequence input mode",
        ["Edit Table", "Custom CSV"],
        index=0,
        horizontal=True,
    )

    def build_default_sequence(df: pd.DataFrame, region: str, crop: str) -> np.ndarray:
        seq = np.zeros((1, 12, len(CLIMATE_FEATURES)), dtype=np.float32)
        if df.empty:
            return seq
        subset = df
        if region in df['Region'].values:
            subset = subset[subset['Region'] == region]
        if crop in df['Crop'].values:
            subset = subset[subset['Crop'] == crop]

        for f_i, feat in enumerate(CLIMATE_FEATURES):
            for m in range(1, 13):
                col = f"{feat}_m{m}"
                if col in subset.columns:
                    seq[0, m - 1, f_i] = float(subset[col].median())
                else:
                    seq[0, m - 1, f_i] = 0.0
        return seq

    X_seq = build_default_sequence(processed_dataset, region, crop)
    baseline_seq = build_global_climatology_sequence(processed_dataset)

    if seq_mode == "Edit Table":
        default_df = pd.DataFrame(X_seq[0], columns=CLIMATE_FEATURES, index=get_month_labels())
        edited_df = st.data_editor(
            default_df,
            width="stretch",
            num_rows="fixed",
            hide_index=False,
            key="climate_sequence_editor",
        )
        try:
            edited_values = edited_df[CLIMATE_FEATURES].to_numpy(dtype=np.float32)
            if edited_values.shape == (12, len(CLIMATE_FEATURES)):
                X_seq[0] = edited_values
            else:
                st.error("The table must contain exactly 12 rows and 9 climate columns.")
                st.stop()
        except Exception as exc:
            st.error(f"Could not read the edited table: {exc}")
            st.stop()
    else:
        st.info(
            "CSV format requirements: the file must contain exactly these 9 columns (headers):\n"
            "`T2M,T2M_MAX,T2M_MIN,TS,T2MDEW,T2MWET,PRECTOTCORR,RH2M,QV2M` and exactly 12 rows (one per month).\n"
            "Do NOT include a separate leading 'Month' column — if your file has month labels, remove that column so the nine climate feature columns are the CSV headers.\n"
            "You can use the provided sample `test.csv` as a reference."
        )
        uploaded = st.file_uploader("Upload CSV with 12 rows and 9 climate columns", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to define the monthly climate sequence.")
            st.stop()

        try:
            csv_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

        # Short validation summary: show detected columns and row count
        try:
            detected_cols = list(csv_df.columns)
            st.info(f"Detected {len(csv_df)} rows and {len(detected_cols)} columns. Columns: {', '.join(detected_cols)}")
        except Exception:
            pass

        missing_cols = [col for col in CLIMATE_FEATURES if col not in csv_df.columns]
        if missing_cols or len(csv_df) != 12:
            st.error("CSV must contain the 9 climate feature columns and exactly 12 rows.")
            st.stop()

        X_seq[0] = csv_df[CLIMATE_FEATURES].to_numpy(dtype=np.float32)

    st.markdown("---")

    # Provide a downloadable CSV template for users
    template_df = pd.DataFrame(np.zeros((12, len(CLIMATE_FEATURES)), dtype=np.float32), columns=CLIMATE_FEATURES)
    csv_template = template_df.to_csv(index=False)
    
    st.info("You can edit the table above or upload a CSV. Use the template if unsure of format.")
    
    # Responsive button layout
    btn_col1, btn_col2 = st.columns(2, gap="small")
    with btn_col1:
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="climate_sequence_template.csv",
            mime="text/csv",
            width='stretch'
        )
        # Provide the cleaned sample test.csv as an alternative example
        try:
            sample_path = Path(__file__).resolve().parent.parent / "test.csv"
            if sample_path.exists():
                with open(sample_path, "rb") as fh:
                    sample_bytes = fh.read()
                st.download_button(
                    label="Download sample CSV (test.csv)",
                    data=sample_bytes,
                    file_name="test.csv",
                    mime="text/csv",
                    width='stretch',
                )
        except Exception:
            pass
    
    with btn_col2:
        generate = st.button(
            "🎯 Generate Prediction",
            type="primary",
            width='content'
        )
        if generate:
            st.session_state['generated_prediction'] = True

        # Allow clearing the generated results
        if st.button("Reset Prediction", type="secondary"):
            st.session_state['generated_prediction'] = False

    # Prediction (real path)
    if st.session_state.get('generated_prediction', False):
        st.info("Loading models and making the prediction. This may take a few seconds.")

        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            st.error(f"Required ML libraries not available: {exc}")
            return

        try:
            artifacts = load_prediction_artifacts()
            loaded_models = load_ensemble_models()
        except Exception as exc:
            st.error(f"Prediction setup failed: {exc}")
            st.code(traceback.format_exc())
            return

        df = processed_dataset.copy() if not processed_dataset.empty else pd.DataFrame()

        x_scaler = artifacts.get("x_scaler")
        year_scaler = artifacts.get("year_scaler")

        if x_scaler is None:
            if not df.empty:
                n_feat = len(CLIMATE_FEATURES)
                X_all = np.zeros((df.shape[0], 12, n_feat), dtype=np.float32)
                for i, feat in enumerate(CLIMATE_FEATURES):
                    for m in range(1, 13):
                        col = f"{feat}_m{m}"
                        if col in df.columns:
                            X_all[:, m - 1, i] = df[col].fillna(df[col].median()).values
                x_scaler = StandardScaler()
                x_scaler.fit(X_all.reshape(-1, n_feat))
            st.warning("Using live-fit feature scaling because x_scaler.pkl is not available.")

        if year_scaler is None:
            if not df.empty:
                yr_all = build_year_features_arr(df["Year"].values.astype(np.float32))
                year_scaler = StandardScaler()
                year_scaler.fit(yr_all)
            st.warning("Using live-fit year scaling because year_scaler.pkl is not available.")

        if x_scaler is not None:
            X_in = x_scaler.transform(X_seq.reshape(-1, len(CLIMATE_FEATURES))).reshape(X_seq.shape).astype(np.float32)
            baseline_in = x_scaler.transform(baseline_seq.reshape(-1, len(CLIMATE_FEATURES))).reshape(baseline_seq.shape).astype(np.float32)
        else:
            X_in = X_seq.astype(np.float32)
            baseline_in = baseline_seq.astype(np.float32)

        yr_in = build_year_features_arr(np.array([year], dtype=np.float32))
        if year_scaler is not None:
            yr_in = year_scaler.transform(yr_in).astype(np.float32)

        try:
            p_idx = CLIMATE_FEATURES.index("PRECTOTCORR")
            precip_seq = X_seq[0, :, p_idx]
            if np.allclose(precip_seq, 0.0):
                force = st.checkbox("Force prediction with zero rainfall (not recommended)", value=False)
                st.error("No rainfall provided in the monthly profile.")
                if not force:
                    st.subheader("Prediction not available for zero rainfall")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted Yield", "0 kg/ha")
                    c2.metric("Lower Bound (95%)", "0 kg/ha")
                    c3.metric("Upper Bound (95%)", "0 kg/ha")
                    st.info("Provide non-zero rainfall values or check the force option to proceed anyway.")
                    return
        except Exception:
            pass

        if artifacts.get("region_to_id") and artifacts.get("crop_to_id"):
            region_to_id = artifacts["region_to_id"]
            crop_to_id = artifacts["crop_to_id"]
            try:
                r_id = int(region_to_id[region])
                c_id = int(crop_to_id[crop])
            except Exception:
                st.error("Selected region/crop not found in saved label mappings.")
                return
        else:
            if not processed_dataset.empty:
                regions_sorted = sorted(processed_dataset["Region"].astype(str).unique())
                crops_sorted = sorted(processed_dataset["Crop"].astype(str).unique())
            else:
                regions_sorted = region_options
                crops_sorted = crop_options

            try:
                r_id = int(regions_sorted.index(region))
                c_id = int(crops_sorted.index(crop))
            except ValueError:
                st.error("Selected region/crop not found in fallback mapping.")
                return

            st.warning("Label mapping file not found; using a simple fallback mapping.")

        try:
            X_tensor = tf.convert_to_tensor(X_in)
            r_tensor = tf.convert_to_tensor(np.array([r_id], dtype=np.int32))
            c_tensor = tf.convert_to_tensor(np.array([c_id], dtype=np.int32))
            yr_tensor = tf.convert_to_tensor(yr_in)

            preds_norm = []
            for model in loaded_models:
                try:
                    outputs = model([X_tensor, r_tensor, c_tensor, yr_tensor], training=False)
                    preds_norm.append(np.asarray(outputs).reshape(-1))
                except Exception as exc:
                    st.warning(f"A model failed during prediction: {exc}")

            if not preds_norm:
                st.error("All models failed during prediction.")
                return

            preds_array = np.stack(preds_norm, axis=1)
            mean_norm = preds_array.mean(axis=1).ravel()[0]
            std_norm = preds_array.std(axis=1, ddof=1).ravel()[0] if preds_array.shape[1] > 1 else 0.0

            crop_stats = artifacts.get("crop_stats")
            crop_mean = 0.0
            crop_std = 1.0

            if isinstance(crop_stats, dict):
                crop_means = crop_stats.get("crop_log_means") or crop_stats.get("crop_means") or {}
                crop_stds = crop_stats.get("crop_log_stds") or crop_stats.get("crop_stds") or {}
                crop_mean = float(crop_means.get(crop, np.mean(list(crop_means.values())) if crop_means else 0.0))
                crop_std = float(crop_stds.get(crop, np.nan))
                if np.isnan(crop_std) or crop_std == 0.0:
                    crop_std = float(np.std(list(crop_means.values()))) if crop_means else 1.0
            elif not df.empty and "Yield_kg_per_ha" in df.columns:
                crop_stats_df = df.copy()
                crop_stats_df["y_log"] = np.log(crop_stats_df["Yield_kg_per_ha"].fillna(0.0) + EPSILON)
                crop_means = crop_stats_df.groupby("Crop")["y_log"].mean().to_dict()
                crop_stds = crop_stats_df.groupby("Crop")["y_log"].std().to_dict()
                crop_mean = float(crop_means.get(crop, np.mean(list(crop_means.values()))))
                crop_std = float(crop_stds.get(crop, np.nan))
                if np.isnan(crop_std) or crop_std == 0.0:
                    crop_std = float(np.std(list(crop_means.values()))) if crop_means else 1.0

            y_log_pred = mean_norm * crop_std + crop_mean
            y_log_lower = (mean_norm - 1.96 * std_norm) * crop_std + crop_mean
            y_log_upper = (mean_norm + 1.96 * std_norm) * crop_std + crop_mean

            y_pred = max(np.exp(y_log_pred) - EPSILON, 0.0)
            y_lower = max(np.exp(y_log_lower) - EPSILON, 0.0)
            y_upper = max(np.exp(y_log_upper) - EPSILON, 0.0)

            half_width = (y_upper - y_lower) / 2.0
            uncertainty_pct = (half_width / max(y_pred, EPSILON)) * 100.0
            
            # Convert uncertainty to confidence (Model Agreement %)
            model_confidence = max(0.0, min(100.0, 100.0 - uncertainty_pct))

            st.subheader("📊 Prediction Results")

            # Explanation of confidence calculation
            with st.expander("ℹ️ How is Model Ensemble Confidence calculated?", expanded=False):
                st.markdown(
                    """
**Definition:**

Model Ensemble Confidence = `100 - uncertainty_pct`, where:
- `uncertainty_pct = (CI_half_width / mean_prediction) * 100`
- `CI_half_width` = half-width of the 95% confidence interval across the 5 ensemble models
- `mean_prediction` = ensemble mean yield estimate

**Interpretation:**

- **90%+**: Excellent. All 5 models strongly agree. Prediction is very reliable for planning.
- **75–90%**: Good. Models mostly agree with minor spread. Suitable for decision-making.
- **60–75%**: Moderate. Noticeable spread; use as one input among others.
- **<60%**: Low. High disagreement among models. Gather additional data or seek expert input.

**What it reflects:**
- How well the ensemble members agree, not absolute accuracy.
- A narrow confidence interval (models agreeing) → high confidence.
- A wide confidence interval (models disagreeing) → low confidence.

**What it does NOT reflect:**
- Whether the model is correct (accuracy vs. test data).
- Bias in all 5 models agreeing on a wrong answer.
- Unrepresented climate scenarios (out-of-training-distribution inputs).

**Recommendation:**
Note: this is about model agreement, not a guarantee the prediction is correct. Combine it with domain knowledge (local agronomist insight, historical context, management constraints) for robust planning.
                    """
                )

            # 1. Expected Range (PRIMARY - most actionable)
            st.metric(
                "🎯 Expected Yield Range (approx.)",
                f"{y_lower:,.0f} – {y_upper:,.0f} kg/ha",
                delta=f"±{half_width:,.0f} kg/ha",
                delta_color="normal"
            )

            # 2. Model Confidence (dynamic, intuitive)
            if model_confidence >= 90:
                confidence_label = "High Confidence 🟢"
                confidence_interpretation = "Models strongly agree — high reliability"
                delta_color = "normal"
            elif model_confidence >= 75:
                confidence_label = "Good Confidence 🟡"
                confidence_interpretation = "Good agreement — suitable for planning"
                delta_color = "normal"
            elif model_confidence >= 60:
                confidence_label = "Moderate Confidence 🟠"
                confidence_interpretation = "Moderate disagreement — use with caution"
                delta_color = "off"
            else:
                confidence_label = "Low Confidence 🔴"
                confidence_interpretation = "High disagreement — seek additional data"
                delta_color = "inverse"

            st.metric(
                "🤝 Model Ensemble Confidence",
                f"{model_confidence:.1f}%",
                delta=confidence_label,
                delta_color=delta_color,
                help="Reflects how much the 5 ensemble models agree with each other. Higher % = stronger consensus.\n\n"
                     "**Calculation**: 100% - (uncertainty spread as %% of prediction).\n"
                     "Uncertainty is derived from 95%% CI half-width across ensemble members.\n"
                     "~90%+: Very reliable; <50%: High disagreement, seek more data."
            )

            # 3. Point Estimate (secondary detail)
            st.metric(
                "📍 Best Single Estimate",
                f"{y_pred:,.0f} kg/ha",
                delta=f"{uncertainty_pct:.1f}% spread",
                delta_color="inverse" if uncertainty_pct > 25 else "off",
                help="The ensemble mean prediction; use the range above for planning"
            )

            # 4. Interpretive uncertainty band
            st.metric(
                "📈 Model Agreement",
                confidence_interpretation,
                help=f"Model agreement level based on {uncertainty_pct:.1f}% relative spread"
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Yield"], y=[y_pred], name="Point Estimate", marker_color="#2E86AB"))
            fig.add_trace(
                go.Scatter(
                    x=["Yield"],
                    y=[y_upper],
                    error_y=dict(type="data", array=[y_upper - y_lower], visible=True),
                    mode="markers",
                    marker_color="#A1C6D4",
                    showlegend=False,
                )
            )
            fig.update_layout(
                title="Yield Prediction with uncertainty range",
                yaxis_title="Yield (kg/ha)",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig, width="stretch")
            # Try to export the prediction chart as PNG for inclusion in reports
            pred_png = _export_prediction_chart_png(y_pred, y_lower, y_upper, fig)

            # Compute and display temporal comparison vs regional baseline
            historical_summary = {}
            try:
                regional_baseline = compute_regional_climate_baseline(df, region)
                anomalies_df = compute_climate_anomalies(X_seq, regional_baseline)
                
                if not anomalies_df.empty:
                    st.subheader("📊 How your months compare to the region")
                    st.caption("Shows how unusual each month is compared with the region's typical climate (standard and percent difference).")
                    
                    with st.expander("📈 View anomalies by feature", expanded=False):
                        # Summary stats
                        extreme_anomalies = anomalies_df[anomalies_df['Z_Score'].abs() > 1.5]
                        if not extreme_anomalies.empty:
                            st.warning(
                                f"⚠️ {len(extreme_anomalies)} month(s) look unusually different from the region's typical climate.\n"
                                "These months may require extra attention when planning."
                            )
                        
                        # Display top positive and negative anomalies
                        top_positive = anomalies_df.nlargest(3, 'Z_Score')[['Feature', 'Month', 'User_Value', 'Baseline_Mean', 'Z_Score', 'Anomaly_Percent']]
                        top_negative = anomalies_df.nsmallest(3, 'Z_Score')[['Feature', 'Month', 'User_Value', 'Baseline_Mean', 'Z_Score', 'Anomaly_Percent']]
                        
                        if not top_positive.empty:
                            st.write("**Top 3 Positive Anomalies** (above baseline):")
                            top_positive_display = top_positive.copy()
                            top_positive_display['Z_Score'] = top_positive_display['Z_Score'].round(2)
                            top_positive_display['Anomaly_Percent'] = top_positive_display['Anomaly_Percent'].round(1)
                            top_positive_display['User_Value'] = top_positive_display['User_Value'].round(2)
                            top_positive_display['Baseline_Mean'] = top_positive_display['Baseline_Mean'].round(2)
                            st.dataframe(top_positive_display, hide_index=True, width='stretch')
                        
                        if not top_negative.empty:
                            st.write("**Top 3 Negative Anomalies** (below baseline):")
                            top_negative_display = top_negative.copy()
                            top_negative_display['Z_Score'] = top_negative_display['Z_Score'].round(2)
                            top_negative_display['Anomaly_Percent'] = top_negative_display['Anomaly_Percent'].round(1)
                            top_negative_display['User_Value'] = top_negative_display['User_Value'].round(2)
                            top_negative_display['Baseline_Mean'] = top_negative_display['Baseline_Mean'].round(2)
                            st.dataframe(top_negative_display, hide_index=True, width='stretch')

                    # Temporal comparison vs historical yield baseline
                    try:
                        hist = compute_historical_yield_baseline(df, region, crop)
                        if hist.get('count', 0) > 0 and hist.get('mean') is not None:
                            # Percent difference and z-score
                            pct_diff = (y_pred - hist['mean']) / max(hist['mean'], EPSILON) * 100.0
                            z_score = (y_pred - hist['mean']) / hist['std'] if hist['std'] and hist['std'] > 0 else None
                            # Percentile rank among historical yields
                            ranks = hist.get('yields', [])
                            if ranks:
                                rank_pct = float(np.sum(np.array(ranks) < y_pred) / len(ranks) * 100.0)
                            else:
                                rank_pct = None

                            st.subheader("📅 Comparison vs Historical Yield")
                            if rank_pct is not None:
                                # Simple human phrasing
                                if rank_pct >= 90:
                                    rank_text = f"(Top {100 - int(rank_pct)}% historically)"
                                elif rank_pct <= 10:
                                    rank_text = f"(Bottom {int(rank_pct)}% historically)"
                                else:
                                    rank_text = f"(Percentile {rank_pct:.0f})"
                            else:
                                rank_text = ""

                            pct_label = f"{pct_diff:+.0f}% vs historical mean"
                            if z_score is not None:
                                z_label = f"Z = {z_score:.2f}"
                            else:
                                z_label = "Z = N/A"

                            col_a, col_b = st.columns([2, 3])
                            with col_a:
                                st.metric("Historical mean yield", f"{hist['mean']:.0f} kg/ha", delta=None)
                            with col_b:
                                st.markdown(f"**Prediction vs historical:** {pct_label} — {z_label} {rank_text}")
                            historical_summary = {
                                "mean": float(hist.get("mean", 0.0)),
                                "std": float(hist.get("std", 0.0)),
                                "count": int(hist.get("count", 0)),
                                "pct_diff": float(pct_diff),
                                "z_score": float(z_score) if z_score is not None else None,
                                "percentile_rank": float(rank_pct) if rank_pct is not None else None,
                            }
                        else:
                            st.info("Historical yield baseline not available for this region/crop.")
                    except Exception as e:
                        st.warning(f"Could not compute historical yield comparison: {e}")
            except Exception as e:
                st.warning(f"Could not compute climate anomalies: {e}")

            # Agronomic risk details and Climate Stress Scorecard removed — model not trained
            # to provide learned agronomic risk attributions. We keep driver sensitivity
            # and temporal comparisons only.

            driver_summary = build_feature_sensitivity_summary(
                models=loaded_models,
                X_input=X_in,
                baseline_scaled_sequence=baseline_in,
                baseline_raw_sequence=baseline_seq,
                region_id=r_id,
                crop_id=c_id,
                year_input=yr_in,
                crop_mean=crop_mean,
                crop_std=crop_std,
                user_sequence=X_seq,
            )
            if not driver_summary.empty:
                st.subheader("🔎 Drivers Behind This Prediction")
                st.caption(
                    "We temporarily replace one feature at a time with its typical value to see how the prediction changes. Use these values to see which factors matter most."
                )

                with st.expander("🛈 Explainability guide — what each column means", expanded=False):
                    st.markdown(
                        """
- **Baseline_Mean**: The climatology value for that feature (median across the processed dataset months), shown in raw units (e.g., mm/month, °C). This is the reference we use when "neutralizing" a feature.
- **Baseline (model input)**: The same climatology values but scaled to the model's feature space; used internally when re-evaluating the model.
- **User_Mean**: The mean of the user's monthly sequence for the feature (raw units).
- **User_vs_Baseline_Delta**: `User_Mean - Baseline_Mean`. Positive means the user's value is above climatology; negative means below.
- **Yield_Impact_kg_ha**: Change in predicted yield (kg/ha) when the feature is replaced by the baseline (calculated as `base_yield - perturbed_yield`). Positive means the user's feature values increased predicted yield relative to climatology.
- **Abs_Impact_kg_ha**: Absolute magnitude of the impact; used for ranking the most influential features.
- **Direction**: Simple sign label: `Positive` if `Yield_Impact_kg_ha >= 0`, otherwise `Negative`.
- **Interpretation**: A short, human-readable explanation of why that feature may increase or decrease yield in this context.

**Method**: This app uses a leave-one-feature-out sensitivity test — for each feature we replace the user's monthly profile with the climatology baseline (scaled for the model), re-run the ensemble, and compute the yield difference. This is fast and input-driven and does not rely on precomputed SHAP files.

**Units (typical)**:
- `PRECTOTCORR` = **mm/month** (total rainfall for the month, not daily average)
- Temperatures = °C
- Humidity = % (RH2M) or g/kg (QV2M)
                        """
                    )

                driver_display = driver_summary.copy()
                driver_display["User_Mean"] = driver_display["User_Mean"].round(3)
                driver_display["Baseline_Mean"] = driver_display["Baseline_Mean"].round(3)
                driver_display["User_vs_Baseline_Delta"] = driver_display["User_vs_Baseline_Delta"].round(3)
                driver_display["Yield_Impact_kg_ha"] = driver_display["Yield_Impact_kg_ha"].round(1)
                driver_display["Abs_Impact_kg_ha"] = driver_display["Abs_Impact_kg_ha"].round(1)
                # Round normalized columns if present
                if "Normalized_Impact_kg_ha" in driver_display.columns:
                    driver_display["Normalized_Impact_kg_ha"] = driver_display["Normalized_Impact_kg_ha"].round(1)
                if "Normalized_Abs_Impact_kg_ha" in driver_display.columns:
                    driver_display["Normalized_Abs_Impact_kg_ha"] = driver_display["Normalized_Abs_Impact_kg_ha"].round(1)

                # Add human-readable interpretation text per feature
                driver_display["Interpretation"] = driver_display.apply(
                    lambda r: get_feature_explanation(r["Feature"], r.get("Direction", "")), axis=1
                )

                sequence_df = pd.DataFrame(X_seq[0], columns=CLIMATE_FEATURES, index=get_month_labels())

                # Inline explainability expander for Yield_Impact_kg_ha
                with st.expander("ⓘ What the feature numbers mean", expanded=False):
                    st.markdown(
                        """
Quick summary:

- Each row shows how much the model's yield estimate would change (in kg/ha) if we replace that one climate feature with its typical historical value.
- Use these numbers to see which features are most important for this prediction.

Important notes:
- These are model-based "what-if" results, not proof of real-world cause and effect.
- Because the model considers all features together, the individual numbers may not add up exactly to the model's total change. Use them for ranking and intuition, not as exact arithmetic.
"""
                    )

                # Sanitize DataFrame.attrs to native Python types to avoid pyarrow/json issues in some environments
                try:
                    if isinstance(driver_display.attrs, dict):
                        for _k, _v in list(driver_display.attrs.items()):
                            # numpy scalars expose .item(); convert them to native types
                            try:
                                if hasattr(_v, "item"):
                                    driver_display.attrs[_k] = _v.item()
                            except Exception:
                                # fallback: leave value as-is
                                pass
                except Exception:
                    pass

                cols_to_show = ["Rank", "Feature", "Parameter", "User_Mean", "Baseline_Mean", "User_vs_Baseline_Delta", "Yield_Impact_kg_ha"]
                if "Normalized_Impact_kg_ha" in driver_display.columns:
                    cols_to_show.append("Normalized_Impact_kg_ha")
                cols_to_show.append("Interpretation")

                st.dataframe(
                    driver_display[cols_to_show],
                    width='stretch',
                    hide_index=True,
                )

                # Friendly attribution notice when the numbers don't add up well
                if 'attribution_diagnostic' in getattr(driver_summary, 'attrs', {}):
                    diag = driver_summary.attrs.get('attribution_diagnostic', {})
                    if not diag.get("is_calibrated", True):
                        st.warning(
                            "The individual feature numbers below are useful for ranking which factors matter most, "
                            f"but they do not add up exactly: the features sum to {diag.get('sum_impacts', 0):.0f} kg/ha while the model's total change is {diag.get('total_delta', 0):.0f} kg/ha ({diag.get('divergence_percent', 0.0):.1f}% difference).\n\n"
                            "This happens because the model looks at all inputs together (features can interact). Treat the per-feature values as relative importance, not as a precise breakdown."
                        )

                # Chart mode toggle: Signed (show +/-) or Absolute (magnitude only)
                chart_mode = st.radio(
                    "Chart mode",
                    ["Signed", "Absolute"],
                    index=0,
                    horizontal=True,
                    key="driver_chart_mode",
                )

                st.markdown("**Legend:** 🟦 helps yield (positive impact), 🟥 hurts yield (negative impact)")

                # Coerce numeric values for plotting and ensure color array length matches
                if chart_mode == "Signed":
                    raw_vals = pd.to_numeric(driver_display["Yield_Impact_kg_ha"], errors='coerce').fillna(0.0).astype(float)
                    signed_colors = np.where(raw_vals >= 0, "#2E86AB", "#D1495B")
                    x_values_plot = raw_vals
                    hover_tpl = "%{y}<br>Signed impact: %{x:.1f} kg/ha<extra></extra>"
                    x_axis_title = "Signed yield impact (kg/ha)"
                    text_vals = raw_vals.round(1)
                else:
                    abs_col = "Normalized_Abs_Impact_kg_ha" if "Normalized_Abs_Impact_kg_ha" in driver_display.columns else "Abs_Impact_kg_ha"
                    raw_vals = pd.to_numeric(driver_display[abs_col], errors='coerce').fillna(0.0).astype(float)
                    signed_colors = np.array(["#2E86AB"] * len(raw_vals), dtype=object)
                    x_values_plot = raw_vals
                    hover_tpl = "%{y}<br>Absolute impact: %{x:.1f} kg/ha<extra></extra>"
                    x_axis_title = "Absolute yield impact (kg/ha)"
                    text_vals = raw_vals.round(1)

                # Plot with numeric labels and clearer margins (show only feature codes on y-axis)
                driver_fig = go.Figure(
                    go.Bar(
                        x=x_values_plot.tolist(),
                        y=list(driver_display["Feature"]),
                        orientation="h",
                        marker_color=signed_colors.tolist(),
                        text=text_vals.tolist(),
                        texttemplate="%{text} kg/ha",
                        textposition="auto",
                        hovertemplate=hover_tpl,
                    )
                )
                driver_fig.update_layout(
                    title="Feature Influence on Yield",
                    xaxis_title=x_axis_title,
                    yaxis_title="",
                    template="plotly_white",
                    height=420,
                    margin=dict(l=140, r=20, t=60, b=20),
                    xaxis=dict(zeroline=True, zerolinecolor="#6b7280"),
                )
                st.plotly_chart(driver_fig, width='stretch')
                # Try to export the driver influence chart as PNG for reports
                driver_png = _export_driver_chart_png(driver_display, chart_mode, x_axis_title)

            # Visual interpretation of confidence level
            if uncertainty_pct < 10.0:
                uncertainty_band = "Low"
                st.success(f"✓ Uncertainty band: {uncertainty_band} ({uncertainty_pct:.1f}%) — models largely agree")
            elif uncertainty_pct < 25.0:
                uncertainty_band = "Medium"
                st.info(f"ℹ Uncertainty band: {uncertainty_band} ({uncertainty_pct:.1f}%) — moderate disagreement between models")
            else:
                uncertainty_band = "High"
                st.warning(f"⚠ Uncertainty band: {uncertainty_band} ({uncertainty_pct:.1f}%) — high disagreement; interpret with caution")

            # Build downloadable Markdown/PDF report after uncertainty_band is defined
            report_markdown = None
            try:
                images = {}
                if 'pred_png' in locals() and pred_png is not None:
                    images['prediction_chart'] = pred_png
                if 'driver_png' in locals() and driver_png is not None:
                    images['driver_chart'] = driver_png

                report_markdown = build_prediction_report_markdown(
                    crop=crop,
                    region=region,
                    year=year,
                    y_pred=y_pred,
                    y_lower=y_lower,
                    y_upper=y_upper,
                    uncertainty_pct=uncertainty_pct,
                    model_confidence=model_confidence,
                    uncertainty_band=uncertainty_band,
                    model_members=len(loaded_models),
                    sequence_df=sequence_df,
                    driver_df=driver_display,
                    historical_summary=historical_summary,
                    anomalies_df=anomalies_df if 'anomalies_df' in locals() else None,
                    images=images if images else None,
                )
            except Exception as exc:
                st.warning(f"Could not build Markdown report: {exc}")

            if report_markdown is not None:
                dl_col_md, dl_col_pdf, dl_col_zip = st.columns([1, 1, 1])
                with dl_col_md:
                    st.download_button(
                        label="Download Markdown report",
                        data=report_markdown.encode("utf-8"),
                        file_name=build_prediction_report_filename(crop, region, year, suffix="md"),
                        mime="text/markdown",
                        width='stretch',
                    )

                # Generate PDF (may fail) and expose in its own column
                report_pdf = None
                try:
                    report_pdf = build_prediction_report_pdf_bytes(report_markdown, images=images if images else None)
                except Exception as exc:
                    with dl_col_pdf:
                        st.warning(f"PDF export failed: {exc}")

                with dl_col_pdf:
                    if report_pdf is not None:
                        st.download_button(
                            label="Download PDF report",
                            data=report_pdf,
                            file_name=build_prediction_report_filename(crop, region, year, suffix="pdf"),
                            mime="application/pdf",
                            width='stretch',
                        )
                    else:
                        st.info("PDF not available; download Markdown or ZIP instead.")

                # Build ZIP containing markdown and PDF (if available) in third column
                try:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        md_name = build_prediction_report_filename(crop, region, year, suffix="md")
                        zf.writestr(md_name, report_markdown.encode("utf-8"))
                        if report_pdf is not None:
                            pdf_name = build_prediction_report_filename(crop, region, year, suffix="pdf")
                            zf.writestr(pdf_name, report_pdf)
                    zip_buf.seek(0)
                    with dl_col_zip:
                        st.download_button(
                            label="Download ZIP (MD + PDF)",
                            data=zip_buf.getvalue(),
                            file_name=build_prediction_report_filename(crop, region, year, suffix="zip"),
                            mime="application/zip",
                            width='stretch',
                        )
                except Exception as exc:
                    with dl_col_zip:
                        st.warning(f"Could not build ZIP export: {exc}")

                st.caption(
                    "Both exports include the input parameters, prediction, temporal comparison, confidence, and model limitations."
                )

            st.markdown("---")
            st.subheader("📈 Prediction Details")
            st.markdown(
                f"""
**Crop**: {crop}  
**Region**: {region}  
**Year**: {year}  
**Model ensemble members**: {len(loaded_models)}  
**Uncertainty (half-width, 95% CI)**: {uncertainty_pct:.1f}%  
**Uncertainty band**: {uncertainty_band}

The prediction uses cached training artifacts when available, falls back only when necessary, runs the ensemble once per model, then denormalizes the result back to yield units.

---

📋 **Your Input Data Summary**

You provided a 12-month climate sequence. Each row represents one calendar month (Jan–Dec) with 9 climate features:
- **Temperatures**: T2M, T2M_MAX, T2M_MIN, TS (all in °C)
- **Precipitation**: PRECTOTCORR (**mm/month** — monthly total, not daily average)
- **Humidity**: RH2M (%), QV2M (g/kg), T2MDEW (°C), T2MWET (°C)

---

| Band | Threshold | Meaning |
|---|---:|---|
| 🟢 Low | < 10% | Ensemble models largely agree — high confidence |
| 🟡 Medium | 10–25% | Moderate disagreement — use with some caution |
| 🔴 High | ≥ 25% | High disagreement — interpret carefully |
                """
            )

        except Exception:
            st.error("Prediction failed.")
            st.code(traceback.format_exc())
            return


if __name__ == "__main__":
    render()
