import streamlit as st
import pandas as pd
import numpy as np
from data_loader import apply_global_style, load_data

# Small constant used in log-transform inversion
EPSILON = 1e-6


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

    def _series_stats(column_name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        if processed_dataset.empty or column_name not in processed_dataset.columns:
            return default

        values = processed_dataset[column_name].dropna()
        if values.empty:
            return default

        return (
            float(values.quantile(0.05)),
            float(values.median()),
            float(values.quantile(0.95)),
        )

    temp_min, temp_default, temp_max = _series_stats("T2M_m1", (23.3, 25.7, 30.5))
    rainfall_min, rainfall_default, rainfall_max = _series_stats("PRECTOTCORR_m1", (0.0, 540.7, 2068.0))
    humidity_min, humidity_default, humidity_max = _series_stats("RH2M_m1", (21.6, 79.9, 90.7))
    
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
    
    st.subheader("🌡️ Climate Reference Inputs")

    st.caption(
        "The trained model uses a 12-month climate sequence. These sliders provide a coarse historical reference profile based on the processed dataset."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.slider(
            "Reference Monthly Mean Temperature (°C)",
            min_value=float(round(temp_min, 1)),
            max_value=float(round(temp_max, 1)),
            value=float(round(temp_default, 1)),
            step=0.5
        )
    
    with col2:
        rainfall = st.slider(
            "Reference Monthly Rainfall (mm)",
            min_value=float(round(rainfall_min, 1)),
            max_value=float(round(rainfall_max, 1)),
            value=float(round(rainfall_default, 1)),
            step=10.0
        )
    
    with col3:
        humidity = st.slider(
            "Reference Relative Humidity (%)",
            min_value=float(round(humidity_min, 1)),
            max_value=float(round(humidity_max, 1)),
            value=float(round(humidity_default, 1)),
            step=1.0
        )

    with st.expander("Why these ranges?", expanded=False):
        st.markdown(
            f"""
These controls are anchored to the processed dataset rather than arbitrary guesses.

- **Year**: {year_min} to {year_max}, set as a future projection window.
- **Temperature**: observed monthly mean-temperature distribution from the dataset.
- **Rainfall**: observed monthly precipitation distribution from the dataset.
- **Humidity**: observed monthly relative humidity distribution from the dataset.

The current page is still a coarse scenario interface. The real network consumes 12 monthly values for 9 climate features, so a full inference form would need monthly sequence inputs rather than a single summary per variable.
            """
        )
    
    st.markdown("---")
    
    # Advanced: build monthly sequences
    st.subheader("🧾 Monthly Climate Sequence (Advanced)")
    seq_mode = st.radio("Sequence input mode", ["Use dataset median profile (recommended)", "Customize monthly profile"], index=0)

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

    if seq_mode.startswith("Customize"):
        st.markdown("Provide monthly values for each climate feature (12 comma-separated numbers). Example: 23.1,23.4,...")
        user_seq = {}
        for feat in CLIMATE_FEATURES:
            default_vals = ",".join([f"{v:.2f}" for v in X_seq[0, :, CLIMATE_FEATURES.index(feat)]])
            text = st.text_area(f"{feat} (m1..m12)", value=default_vals, help=f"Comma-separated 12 monthly values for {feat}")
            try:
                arr = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
                if len(arr) == 12:
                    user_seq[feat] = arr
                else:
                    st.error(f"{feat}: expected 12 values, got {len(arr)}")
            except Exception:
                st.error(f"{feat}: could not parse numbers")

        if len(user_seq) == len(CLIMATE_FEATURES):
            for f_i, feat in enumerate(CLIMATE_FEATURES):
                X_seq[0, :, f_i] = np.array(user_seq[feat], dtype=np.float32)

    st.markdown("---")

    # Prediction (real path)
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        st.info("Running full inference pipeline: preprocessing, model loading, ensemble prediction, denormalization.")

        try:
            import tensorflow as tf
            from tensorflow import keras
            from sklearn.preprocessing import StandardScaler
        except Exception as e:
            st.error(f"Required ML libraries not available: {e}")
            return

        # Fit full-data scalers on processed dataset (note: differs from fold scalers)
        df = processed_dataset.copy() if not processed_dataset.empty else pd.DataFrame()
        if not df.empty:
            N = df.shape[0]
            n_feat = len(CLIMATE_FEATURES)
            X_all = np.zeros((N, 12, n_feat), dtype=np.float32)
            for i, feat in enumerate(CLIMATE_FEATURES):
                for m in range(1, 13):
                    col = f"{feat}_m{m}"
                    if col in df.columns:
                        X_all[:, m - 1, i] = df[col].fillna(df[col].median()).values
                    else:
                        X_all[:, m - 1, i] = 0.0
            x_scaler = StandardScaler()
            x_scaler.fit(X_all.reshape(-1, n_feat))
        else:
            x_scaler = None

        def build_year_features_arr(years: np.ndarray) -> np.ndarray:
            y_norm = ((years.reshape(-1, 1) - 1999.0) / 24.0).astype(np.float32)
            y_sin = np.sin(2.0 * np.pi * y_norm).astype(np.float32)
            y_cos = np.cos(2.0 * np.pi * y_norm).astype(np.float32)
            return np.column_stack([y_norm, y_sin, y_cos]).astype(np.float32)

        if not df.empty:
            yr_all = build_year_features_arr(df['Year'].values.astype(np.float32))
            year_scaler = StandardScaler()
            year_scaler.fit(yr_all)
        else:
            year_scaler = None

        # Scale inputs
        if x_scaler is not None:
            X_in = x_scaler.transform(X_seq.reshape(-1, len(CLIMATE_FEATURES))).reshape(X_seq.shape).astype(np.float32)
        else:
            X_in = X_seq.astype(np.float32)

        yr_in = build_year_features_arr(np.array([year], dtype=np.float32))
        if year_scaler is not None:
            yr_in = year_scaler.transform(yr_in).astype(np.float32)

        # Map region/crop ids as sorted lists (matches training notebook)
        if not processed_dataset.empty:
            regions_sorted = sorted(processed_dataset['Region'].astype(str).unique())
            crops_sorted = sorted(processed_dataset['Crop'].astype(str).unique())
        else:
            regions_sorted = region_options
            crops_sorted = crop_options

        try:
            r_id = int(regions_sorted.index(region))
            c_id = int(crops_sorted.index(crop))
        except ValueError:
            st.error("Selected region/crop not found in processed dataset mapping.")
            return

        # Discover model files
        model_paths = []
        try:
            inv = pd.read_csv('results/tcn_mlp_model_inventory.csv')
            import os
            for p in inv['Path'].dropna().tolist():
                cleaned = p.replace('..' + os.sep, '').replace('../', '')
                model_paths.append(cleaned)
        except Exception:
            import os
            models_dir = 'models'
            if os.path.isdir(models_dir):
                for f in os.listdir(models_dir):
                    if f.lower().endswith('.keras') or f.lower().endswith('.h5'):
                        model_paths.append(os.path.join(models_dir, f))

        loaded_models = []
        for mp in model_paths:
            try:
                mpath = mp if mp.startswith('models') or mp.startswith('/') else mp
                model = keras.models.load_model(mpath, compile=False)
                loaded_models.append(model)
            except Exception:
                try:
                    import os
                    mpath2 = os.path.join(os.getcwd(), mp)
                    model = keras.models.load_model(mpath2, compile=False)
                    loaded_models.append(model)
                except Exception:
                    continue

        if not loaded_models:
            st.warning("No trained model files found; showing illustrative prediction instead.")
            predicted_yield = np.random.normal(2500, 300)
            confidence_lower = predicted_yield - 200
            confidence_upper = predicted_yield + 200
            st.metric("Predicted Yield", f"{predicted_yield:.0f} kg/ha")
            st.metric("Lower Bound (95%)", f"{confidence_lower:.0f} kg/ha")
            st.metric("Upper Bound (95%)", f"{confidence_upper:.0f} kg/ha")
            return

        # Run predictions across ensemble
        preds_norm = []
        for model in loaded_models:
            try:
                p = model.predict([X_in, np.array([[r_id]]), np.array([[c_id]]), yr_in], verbose=0).ravel()
                preds_norm.append(p)
            except Exception as e:
                st.warning(f"A model failed during prediction: {e}")

        if not preds_norm:
            st.error("All models failed during prediction.")
            return

        preds_array = np.stack(preds_norm, axis=1)
        mean_norm = preds_array.mean(axis=1).ravel()[0]
        std_norm = preds_array.std(axis=1, ddof=1).ravel()[0] if preds_array.shape[1] > 1 else 0.0

        # De-normalize using per-crop log-yield mean/std computed from processed dataset
        if not processed_dataset.empty and 'Yield_kg_per_ha' in processed_dataset.columns:
            crop_stats = processed_dataset.copy()
            crop_stats['y_log'] = np.log(crop_stats['Yield_kg_per_ha'].fillna(0.0) + 1e-6)
            crop_means = crop_stats.groupby('Crop')['y_log'].mean().to_dict()
            crop_stds = crop_stats.groupby('Crop')['y_log'].std().to_dict()
            crop_mean = float(crop_means.get(crop, np.mean(list(crop_means.values()))))
            crop_std = float(crop_stds.get(crop, np.nan))
            if np.isnan(crop_std) or crop_std == 0.0:
                crop_std = float(np.std(list(crop_means.values()))) if crop_means else 1.0
        else:
            crop_mean = 0.0
            crop_std = 1.0

        y_log_pred = mean_norm * crop_std + crop_mean
        y_log_lower = (mean_norm - 1.96 * std_norm) * crop_std + crop_mean
        y_log_upper = (mean_norm + 1.96 * std_norm) * crop_std + crop_mean

        y_pred = max(np.exp(y_log_pred) - EPSILON, 0.0)
        y_lower = max(np.exp(y_log_lower) - EPSILON, 0.0)
        y_upper = max(np.exp(y_log_upper) - EPSILON, 0.0)

        st.subheader("📊 Prediction Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Yield", f"{y_pred:.0f} kg/ha")
        c2.metric("Lower Bound (95%)", f"{y_lower:.0f} kg/ha")
        c3.metric("Upper Bound (95%)", f"{y_upper:.0f} kg/ha")

        st.markdown("---")
        st.subheader("📈 Prediction Details")
        st.markdown(f"""
        **Crop**: {crop}  
        **Region**: {region}  
        **Year**: {year}  
        **Model ensemble members**: {len(loaded_models)}  
        **Ensemble std (normalized space)**: {std_norm:.4f}
        
        The prediction was generated by: preprocessing the custom/historical monthly sequence, scaling with full-dataset scalers fitted on the processed dataset, running each ensemble member, averaging normalized outputs, then de-normalizing using per-crop log-yield mean/std and inverting the log transform.
        """)


if __name__ == "__main__":
    render()
