from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

EPSILON = 1e-6

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
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


def compute_regional_climate_baseline(df: pd.DataFrame, region: str) -> dict:
    baseline_stats = {}
    if df.empty or region not in df["Region"].values:
        return baseline_stats

    region_df = df[df["Region"] == region]
    for feature in CLIMATE_FEATURES:
        for month in range(1, 13):
            col_name = f"{feature}_m{month}"
            if col_name in region_df.columns:
                col_data = region_df[col_name].dropna()
                if len(col_data) > 0:
                    baseline_stats[col_name] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std(ddof=1)) if len(col_data) > 1 else 0.0,
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                    }
    return baseline_stats


def compute_climate_anomalies(user_sequence: np.ndarray, baseline_stats: dict) -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        for feature_idx, feature_name in enumerate(CLIMATE_FEATURES):
            col_key = f"{feature_name}_m{month}"
            if col_key in baseline_stats:
                user_val = float(user_sequence[0, month - 1, feature_idx])
                stats = baseline_stats[col_key]
                baseline_mean = stats["mean"]
                baseline_std = stats["std"]

                if baseline_std > 0:
                    z_score = (user_val - baseline_mean) / baseline_std
                else:
                    z_score = 0.0

                if baseline_mean != 0:
                    anomaly_pct = ((user_val - baseline_mean) / abs(baseline_mean)) * 100.0
                else:
                    anomaly_pct = 0.0

                rows.append(
                    {
                        "Feature": feature_name,
                        "Month": month,
                        "User_Value": user_val,
                        "Baseline_Mean": baseline_mean,
                        "Z_Score": z_score,
                        "Anomaly_Percent": anomaly_pct,
                    }
                )

    return pd.DataFrame(rows)


def compute_agronomic_risks(user_sequence: np.ndarray, crop: str) -> list[dict]:
    risks = []

    precip_seq = user_sequence[0, :, CLIMATE_FEATURES.index("PRECTOTCORR")]
    t2m_seq = user_sequence[0, :, CLIMATE_FEATURES.index("T2M")]
    t2m_max_seq = user_sequence[0, :, CLIMATE_FEATURES.index("T2M_MAX")]
    t2m_min_seq = user_sequence[0, :, CLIMATE_FEATURES.index("T2M_MIN")]
    rh2m_seq = user_sequence[0, :, CLIMATE_FEATURES.index("RH2M")]

    if crop in ["Cassava", "Yam"]:
        critical_months = [6, 7, 8, 9]
        high_precip_months = [m for m in critical_months if m <= 12 and precip_seq[m - 1] > 300]
        if len(high_precip_months) >= 2:
            risks.append(
                {
                    "risk_type": "💧 Waterlogging Risk",
                    "severity": "HIGH" if len(high_precip_months) >= 3 else "MODERATE",
                    "months_affected": high_precip_months,
                    "reason": f"{len(high_precip_months)} months with >300 mm rainfall during root bulking (months 6-9)",
                    "recommendation": "Consider ridge/mound planting and improved soil drainage. Early harvesting may be needed if waterlogging persists.",
                }
            )
    elif crop == "Maize":
        critical_months = [7, 8, 9]
        high_precip_months = [m for m in critical_months if m <= 12 and precip_seq[m - 1] > 250]
        if len(high_precip_months) >= 2:
            risks.append(
                {
                    "risk_type": "💧 Waterlogging Risk",
                    "severity": "MODERATE" if len(high_precip_months) == 2 else "HIGH",
                    "months_affected": high_precip_months,
                    "reason": f"{len(high_precip_months)} months with >250 mm rainfall during grain fill (months 7-9)",
                    "recommendation": "Ensure adequate field drainage. Consider earlier planting to shift risk window.",
                }
            )

    heat_stress_months = [m for m in range(1, 13) if t2m_max_seq[m - 1] > 35]
    if len(heat_stress_months) >= 2:
        risks.append(
            {
                "risk_type": "🔥 Heat Stress Risk",
                "severity": "HIGH" if any(t2m_max_seq[m - 1] > 38 for m in heat_stress_months) else "MODERATE",
                "months_affected": heat_stress_months,
                "reason": f"{len(heat_stress_months)} months with T_max > 35°C (pollination & grain fill sensitive)",
                "recommendation": "Ensure adequate soil moisture. Consider shade management or drought-tolerant varieties. Monitor for heat-induced sterility.",
            }
        )

    high_humid_warm_nights = []
    for m in range(1, 13):
        if rh2m_seq[m - 1] > 80 and t2m_min_seq[m - 1] > 20:
            high_humid_warm_nights.append(m)

    if len(high_humid_warm_nights) >= 3:
        risks.append(
            {
                "risk_type": "🦠 Disease Pressure (High)",
                "severity": "MODERATE" if len(high_humid_warm_nights) < 5 else "HIGH",
                "months_affected": high_humid_warm_nights,
                "reason": f"{len(high_humid_warm_nights)} months with RH >80% + T_min >20°C (fungal disease favorable)",
                "recommendation": "Scout for early disease signs (powdery mildew, early/late blight). Improve canopy ventilation; consider fungicide application if disease observed.",
            }
        )

    critical_growth_months = [5, 6, 7, 8]
    low_precip_critical = [m for m in critical_growth_months if m <= 12 and precip_seq[m - 1] < 50]
    if len(low_precip_critical) >= 2:
        risks.append(
            {
                "risk_type": "🌵 Dry Stress Risk",
                "severity": "HIGH" if len(low_precip_critical) >= 3 else "MODERATE",
                "months_affected": low_precip_critical,
                "reason": f"{len(low_precip_critical)} months with <50 mm rainfall during growing season",
                "recommendation": "Plan supplementary irrigation if available. Consider drought-tolerant varieties. Mulch to retain soil moisture.",
            }
        )

    return risks


def compute_climate_stress_score(risks: list[dict]) -> dict:
    high_count = sum(1 for r in risks if r["severity"] == "HIGH")
    moderate_count = sum(1 for r in risks if r["severity"] == "MODERATE")
    low_count = sum(1 for r in risks if r["severity"] == "LOW")

    score = min(10.0, high_count * 2.5 + moderate_count * 1.2 + low_count * 0.3)

    if score < 2:
        category = "🟢 Low Risk"
        interpretation = "Excellent climate conditions. Proceed with planting as planned."
    elif score < 4:
        category = "🟡 Manageable"
        interpretation = "Minor climate stresses. Monitor and implement standard best practices."
    elif score < 6:
        category = "🟠 Moderate Risk"
        interpretation = "Significant climate challenges. Plan adaptive management strategies."
    elif score < 8:
        category = "🔴 High Risk"
        interpretation = "Severe climate stress. Consider crop variety change or adjusted planting date."
    else:
        category = "⚫ Extreme Risk"
        interpretation = "Extreme conditions detected. Strongly recommend agronomist consultation before planting."

    return {
        "score": score,
        "category": category,
        "interpretation": interpretation,
        "risk_breakdown": f"HIGH: {high_count}, MODERATE: {moderate_count}",
    }


def compute_historical_yield_baseline(df: pd.DataFrame, region: str, crop: str) -> dict:
    result = {"mean": None, "std": None, "count": 0, "years": [], "yields": []}
    if df is None or df.empty:
        return result

    if "Yield_kg_per_ha" not in df.columns:
        return result

    sub = df.copy()
    if region is not None:
        sub = sub[sub["Region"] == region]
    if crop is not None:
        sub = sub[sub["Crop"] == crop]

    if sub.empty:
        return result

    y = sub["Yield_kg_per_ha"].dropna().astype(float)
    if y.empty:
        return result

    result["mean"] = float(y.mean())
    result["std"] = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    result["count"] = int(len(y))
    if "Year" in sub.columns:
        result["years"] = sub["Year"].dropna().astype(int).astype(str).tolist()
    result["yields"] = [float(v) for v in y.tolist()]
    return result


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

    baseline_yield, _, _, _, _ = predict_ensemble_yield(
        models=models,
        X_input=baseline_scaled_sequence,
        region_id=region_id,
        crop_id=crop_id,
        year_input=year_input,
        crop_mean=crop_mean,
        crop_std=crop_std,
    )
    total_delta_expected = base_yield - baseline_yield

    rows = []
    for feature_index, feature_name in enumerate(CLIMATE_FEATURES):
        perturbed_input = X_input.copy()
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

    sum_impacts = summary_df["Yield_Impact_kg_ha"].sum()
    if abs(total_delta_expected) > 0.1:
        divergence_pct = abs(sum_impacts - total_delta_expected) / abs(total_delta_expected) * 100.0
    else:
        divergence_pct = 0.0

    summary_df.attrs["attribution_diagnostic"] = {
        "sum_impacts": float(sum_impacts) if sum_impacts is not None else 0.0,
        "total_delta": float(total_delta_expected) if total_delta_expected is not None else 0.0,
        "divergence_percent": float(divergence_pct),
        "is_calibrated": bool(divergence_pct < 20.0),
    }

    try:
        if abs(float(sum_impacts)) > 0.0:
            factor = float(total_delta_expected) / float(sum_impacts) if abs(float(sum_impacts)) > 1e-9 else 1.0
        else:
            factor = 1.0
    except Exception:
        factor = 1.0

    summary_df["Normalized_Impact_kg_ha"] = (summary_df["Yield_Impact_kg_ha"].astype(float) * factor).astype(float)
    summary_df["Normalized_Abs_Impact_kg_ha"] = summary_df["Normalized_Impact_kg_ha"].abs().astype(float)

    summary_df = summary_df.sort_values(
        by=["Abs_Impact_kg_ha", "Yield_Impact_kg_ha"],
        ascending=False,
    ).reset_index(drop=True)
    summary_df.insert(0, "Rank", np.arange(1, len(summary_df) + 1))
    return summary_df
