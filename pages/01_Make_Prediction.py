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
    st.subheader("🧾 Monthly Climate Sequence (Advanced)")
    with st.expander("📚 Climate Parameter Reference", expanded=False):
        st.code(
            """Category,       Parameter_Code, Parameter_Name,         Unit
temperature,    T2M,            Mean temperature at 2m, °C
temperature,    T2M_MAX,        Max temperature at 2m,  °C
temperature,    T2M_MIN,        Min temperature at 2m,  °C
temperature,    TS,             land surface temperature,°C
rainfall,       PRECTOTCORR,    Bias-corrected total precipitation,mm/day
humidity,       RH2M,           Relative humidity at 2m,%
humidity,       QV2M,           Specific humidity at 2m,g/kg
humidity,       T2MDEW,         Dew point temperature at 2m,°C
humidity,       T2MWET,         Wet bulb temperature at 2m,°C"""
        )

    seq_mode = st.radio(
        "Sequence input mode",
        ["Edit Table", "Custom CSV"],
        index=0,
        horizontal=True,
    )

    CLIMATE_FEATURES = [
        'T2M', 'T2M_MAX', 'T2M_MIN', 'TS', 'T2MDEW', 'T2MWET', 'PRECTOTCORR', 'RH2M', 'QV2M'
    ]

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
        uploaded = st.file_uploader("Upload CSV with 12 rows and 9 climate columns", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to define the monthly climate sequence.")
            st.stop()

        try:
            csv_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

        missing_cols = [col for col in CLIMATE_FEATURES if col not in csv_df.columns]
        if missing_cols or len(csv_df) != 12:
            st.error("CSV must contain the 9 climate feature columns and exactly 12 rows.")
            st.stop()

        X_seq[0] = csv_df[CLIMATE_FEATURES].to_numpy(dtype=np.float32)

    st.markdown("---")

    # Prediction (real path)
    if st.button("🚀 Generate Prediction", type="primary", width='stretch'):
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
        else:
            X_in = X_seq.astype(np.float32)

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

            st.subheader("📊 Prediction Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Yield", f"{y_pred:.0f} kg/ha")
            c2.metric("Lower Bound (95%)", f"{y_lower:.0f} kg/ha")
            c3.metric("Upper Bound (95%)", f"{y_upper:.0f} kg/ha")

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

            st.markdown(f"**Uncertainty (±, 95% CI) — half-width:** {uncertainty_pct:.1f}%")

            if uncertainty_pct < 10.0:
                uncertainty_band = "Low"
                st.success(f"Uncertainty band: {uncertainty_band} ({uncertainty_pct:.1f}%) — models largely agree")
            elif uncertainty_pct < 25.0:
                uncertainty_band = "Medium"
                st.warning(f"Uncertainty band: {uncertainty_band} ({uncertainty_pct:.1f}%) — moderate disagreement between models")
            else:
                uncertainty_band = "High"
                st.error(f"Uncertainty band: {uncertainty_band} ({uncertainty_pct:.1f}%) — high disagreement; interpret with caution")

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
        """
            )

        except Exception:
            st.error("Prediction failed.")
            st.code(traceback.format_exc())
            return


if __name__ == "__main__":
    render()
