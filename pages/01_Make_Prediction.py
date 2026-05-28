import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import traceback
from pathlib import Path

import joblib
import plotly.graph_objects as go

from data_loader import apply_global_style, load_data

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


def get_feature_explanation(feature: str, direction: str) -> str:
    base = FEATURE_EXPLANATIONS.get(feature, "A climate feature affecting crop growth.")
    if pd.isna(direction):
        return base
    direction_text = "increases" if direction.lower().startswith("p") or direction.lower() == "positive" else "decreases"
    return f"{base} In this prediction, neutralizing this feature {direction_text} predicted yield."


def get_month_labels() -> list[str]:
    return [f"Month {month}" for month in range(1, 13)]


def get_os_identifier() -> str:
    return "windows" if os.name == "nt" else "linux"


def normalize_model_path(path_value: str) -> str:
    normalized = str(path_value).strip().replace("../", "").replace("..\\", "").replace("\\", "/")
    if get_os_identifier() == "windows":
        return normalized.replace("/", "\\")
    return normalized


@st.cache_resource(show_spinner="Loading prediction artifacts...")
def load_prediction_artifacts() -> dict:
    artifacts = {
        "x_scaler": None,
        "year_scaler": None,
        "crop_stats": None,
        "region_to_id": None,
        "crop_to_id": None,
    }

    if X_SCALER_PATH.exists():
        try:
            artifacts["x_scaler"] = joblib.load(X_SCALER_PATH)
        except Exception:
            artifacts["x_scaler"] = None

    if YEAR_SCALER_PATH.exists():
        try:
            artifacts["year_scaler"] = joblib.load(YEAR_SCALER_PATH)
        except Exception:
            artifacts["year_scaler"] = None

    if CROP_STATS_PATH.exists():
        try:
            artifacts["crop_stats"] = joblib.load(CROP_STATS_PATH)
        except Exception:
            artifacts["crop_stats"] = None

    if LABEL_MAPPINGS_PATH.exists():
        try:
            with open(LABEL_MAPPINGS_PATH, "r", encoding="utf-8") as handle:
                mappings = json.load(handle)
            artifacts["region_to_id"] = mappings.get("region_to_id")
            artifacts["crop_to_id"] = mappings.get("crop_to_id")
        except Exception:
            artifacts["region_to_id"] = None
            artifacts["crop_to_id"] = None

    return artifacts


@st.cache_resource(show_spinner="Loading ensemble models...")
def load_ensemble_models() -> list:
    models = []

    if MODEL_DIR.is_dir():
        for model_path in sorted(MODEL_DIR.glob("tcn_mlp_fold_*.keras")):
            try:
                import tensorflow as tf

                models.append(tf.keras.models.load_model(str(model_path), compile=False))
            except Exception:
                continue

    if not models:
        fallback_primary = MODEL_DIR / "TCN_MLP_ENSEMBLE_best_fold.keras"
        if fallback_primary.exists():
            import tensorflow as tf

            models.append(tf.keras.models.load_model(str(fallback_primary), compile=False))

    if not models:
        raise FileNotFoundError("No trained ensemble models found in /models")

    return models


def build_year_features_arr(years: np.ndarray) -> np.ndarray:
    y_norm = ((years.reshape(-1, 1) - 1999.0) / 24.0).astype(np.float32)
    y_sin = np.sin(2.0 * np.pi * y_norm).astype(np.float32)
    y_cos = np.cos(2.0 * np.pi * y_norm).astype(np.float32)
    return np.column_stack([y_norm, y_sin, y_cos]).astype(np.float32)


def predict_ensemble_yield(
    models: list,
    X_input: np.ndarray,
    region_id: int,
    crop_id: int,
    year_input: np.ndarray,
    crop_mean: float,
    crop_std: float,
) -> tuple[float, float, float, float, float]:
    import tensorflow as tf

    X_tensor = tf.convert_to_tensor(X_input)
    r_tensor = tf.convert_to_tensor(np.array([region_id], dtype=np.int32))
    c_tensor = tf.convert_to_tensor(np.array([crop_id], dtype=np.int32))
    yr_tensor = tf.convert_to_tensor(year_input)

    preds_norm = []
    for model in models:
        try:
            outputs = model([X_tensor, r_tensor, c_tensor, yr_tensor], training=False)
            preds_norm.append(np.asarray(outputs).reshape(-1))
        except Exception:
            continue

    if not preds_norm:
        raise RuntimeError("All models failed during prediction.")

    preds_array = np.stack(preds_norm, axis=1)
    mean_norm = preds_array.mean(axis=1).ravel()[0]
    std_norm = preds_array.std(axis=1, ddof=1).ravel()[0] if preds_array.shape[1] > 1 else 0.0

    y_log_pred = mean_norm * crop_std + crop_mean
    y_log_lower = (mean_norm - 1.96 * std_norm) * crop_std + crop_mean
    y_log_upper = (mean_norm + 1.96 * std_norm) * crop_std + crop_mean

    y_pred = max(np.exp(y_log_pred) - EPSILON, 0.0)
    y_lower = max(np.exp(y_log_lower) - EPSILON, 0.0)
    y_upper = max(np.exp(y_log_upper) - EPSILON, 0.0)

    return y_pred, y_lower, y_upper, mean_norm, std_norm


def build_global_climatology_sequence(df: pd.DataFrame) -> np.ndarray:
    seq = np.zeros((1, 12, len(CLIMATE_FEATURES)), dtype=np.float32)
    if df.empty:
        return seq

    for feature_index, feature_name in enumerate(CLIMATE_FEATURES):
        for month in range(1, 13):
            column_name = f"{feature_name}_m{month}"
            if column_name in df.columns:
                seq[0, month - 1, feature_index] = float(df[column_name].median())

    return seq


def build_feature_sensitivity_summary(
    models: list,
    X_input: np.ndarray,
    baseline_scaled_sequence: np.ndarray,
    baseline_raw_sequence: np.ndarray,
    region_id: int,
    crop_id: int,
    year_input: np.ndarray,
    crop_mean: float,
    crop_std: float,
    user_sequence: np.ndarray,
) -> pd.DataFrame:
    base_yield, _, _, _, _ = predict_ensemble_yield(
        models=models,
        X_input=X_input,
        region_id=region_id,
        crop_id=crop_id,
        year_input=year_input,
        crop_mean=crop_mean,
        crop_std=crop_std,
    )

    rows = []
    for feature_index, feature_name in enumerate(CLIMATE_FEATURES):
        perturbed_input = X_input.copy()
        # use the scaled baseline values for model input
        perturbed_input[0, :, feature_index] = baseline_scaled_sequence[0, :, feature_index]

        perturbed_yield, _, _, _, _ = predict_ensemble_yield(
            models=models,
            X_input=perturbed_input,
            region_id=region_id,
            crop_id=crop_id,
            year_input=year_input,
            crop_mean=crop_mean,
            crop_std=crop_std,
        )

        impact = float(base_yield - perturbed_yield)
        absolute_impact = abs(impact)
        # user_sequence is in raw units; baseline_raw_sequence is also raw units
        user_mean = float(np.mean(user_sequence[0, :, feature_index]))
        baseline_mean = float(np.mean(baseline_raw_sequence[0, :, feature_index]))

        rows.append(
            {
                "Feature": feature_name,
                "Parameter": CLIMATE_FEATURE_LABELS.get(feature_name, feature_name),
                "User_Mean": user_mean,
                "Baseline_Mean": baseline_mean,
                "User_vs_Baseline_Delta": user_mean - baseline_mean,
                "Yield_Impact_kg_ha": impact,
                "Abs_Impact_kg_ha": absolute_impact,
                "Direction": "Positive" if impact >= 0 else "Negative",
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values(
        by=["Abs_Impact_kg_ha", "Yield_Impact_kg_ha"],
        ascending=False,
    ).reset_index(drop=True)
    summary_df.insert(0, "Rank", np.arange(1, len(summary_df) + 1))
    return summary_df


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

1. **Temperature features** (T2M, T2M_MAX, T2M_MIN, TS, T2MDEW, T2MWET): Provide the monthly average or aggregated value from daily observations (°C).
2. **Precipitation (PRECTOTCORR)**: Provide the **monthly total** in mm. If you have daily data, sum all days in the month.
3. **Humidity features** (RH2M, QV2M): Provide the monthly average (% or g/kg).
4. **All 12 rows**: One row per month (Jan–Dec), in order. Your sequence should represent a complete annual climate profile.

**⚠️ Important**: Do not provide daily averages for monthly fields. Use monthly aggregates.
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

    # Prediction (real path)
    if generate:
        st.info("Running cached artifact loading, ensemble prediction, and denormalization.")

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

            st.warning("Using fallback sorted label mapping because label_mappings.json is not available.")

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

            # 1. Expected Range (PRIMARY - most actionable)
            st.metric(
                "🎯 Expected Yield Range (95% Confidence Interval)",
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
                help="Reflects how much the 5 ensemble models agree with each other. Higher % = stronger consensus."
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
                title="Yield Prediction with 95% Confidence Interval",
                yaxis_title="Yield (kg/ha)",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig, width="stretch")

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
                    "This is a fast what-if sensitivity analysis: each feature is neutralized to the climatology baseline one at a time, then the model is re-evaluated to estimate its yield impact."
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

                # Add human-readable interpretation text per feature
                driver_display["Interpretation"] = driver_display.apply(
                    lambda r: get_feature_explanation(r["Feature"], r.get("Direction", "")), axis=1
                )

                # Inline explainability expander for Yield_Impact_kg_ha
                with st.expander("ⓘ Yield_Impact_kg_ha — meaning & caveats", expanded=False):
                    st.markdown(
                        """
**Quick summary (friendly)**

- `Yield_Impact_kg_ha` estimates how many kg/ha the model's predicted yield would change if we replaced one climate feature with its historical climatology. Use it to rank influential features and get intuition about drivers.

**Details (technical)**

- **What it is**: The change in predicted yield (kg/ha) when the user's monthly profile for a single feature is replaced by the climatology baseline and the model is re-evaluated.
- **How to read it**: Positive means the user's current values *increase* predicted yield vs climatology; negative means they *decrease* predicted yield.
- **Why this is not a causal proof**: This is a model-based what-if. It does not prove real-world causation because the model may encode correlations, features co-vary, there may be confounders, and the result depends on model validity.

For causal evidence you'd need randomized interventions or causal-inference methods. Treat this number as a fast, practical interpretation aid — helpful for ranking and scenario comparison, not for proving causation.
                        """
                    )

                st.dataframe(
                    driver_display[["Rank", "Feature", "Parameter", "User_Mean", "Baseline_Mean", "User_vs_Baseline_Delta", "Yield_Impact_kg_ha", "Interpretation"]],
                    width='stretch',
                    hide_index=True,
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

                if chart_mode == "Signed":
                    signed_colors = np.where(driver_display["Yield_Impact_kg_ha"] >= 0, "#2E86AB", "#D1495B")
                    x_values = driver_display["Yield_Impact_kg_ha"]
                    hover_tpl = "%{y}<br>Signed impact: %{x:.1f} kg/ha<extra></extra>"
                    x_axis_title = "Signed yield impact (kg/ha)"
                else:
                    signed_colors = np.where(driver_display["Abs_Impact_kg_ha"] >= 0, "#2E86AB", "#2E86AB")
                    x_values = driver_display["Abs_Impact_kg_ha"]
                    hover_tpl = "%{y}<br>Absolute impact: %{x:.1f} kg/ha<extra></extra>"
                    x_axis_title = "Absolute yield impact (kg/ha)"

                # Plot with numeric labels and clearer margins (show only feature codes on y-axis)
                driver_fig = go.Figure(
                    go.Bar(
                        x=x_values,
                        y=list(driver_display["Feature"]),
                        orientation="h",
                        marker_color=signed_colors,
                        text=x_values.round(1),
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
- **Temperatures**: T2M, T2M_MAX, T2M_MIN, TS, T2MDEW, T2MWET (all in °C)
- **Precipitation**: PRECTOTCORR (**mm/month** — monthly total, not daily average)
- **Humidity**: RH2M (%), QV2M (g/kg)

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
