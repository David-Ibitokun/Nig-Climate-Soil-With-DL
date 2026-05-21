from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


_ROOT = Path(__file__).resolve().parent
_RESULTS_DIR = _ROOT / "results"
_RESULTS_DIR_ALT = _ROOT / "results"  # Fallback to same location
_MODELS_DIR = _ROOT / "models"


def _png_is_mostly_blank(path: Path) -> bool:
    """True if the PNG is effectively a flat white canvas (broken export)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((160, 160), Image.Resampling.LANCZOS)
            pixels = im.getdata()
            n = 0
            s = 0
            for r, g, b in pixels:
                s += r + g + b
                n += 3
            if n == 0:
                return True
            return (s / n) > 254.5
    except Exception:
        return True


def resolve_results_png(filename: str) -> Path:
    """
    Prefer real figures under New_Changes/results, but fall back to yield_changes/results
    when the primary file is missing or an all-white placeholder.
    """
    bases = (_RESULTS_DIR, _RESULTS_DIR_ALT)
    for base in bases:
        p = base / filename
        if p.exists() and not _png_is_mostly_blank(p):
            return p
    for base in bases:
        p = base / filename
        if p.exists():
            return p
    return _RESULTS_DIR / filename


def apply_global_style() -> None:
    st.set_page_config(
        page_title="Climate Change & Food Security (Nigeria)",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
          .stMetric { border: 1px solid rgba(0,0,0,0.06);
                     padding: 0.75rem 0.9rem; border-radius: 0.75rem; }
                    [data-testid="stSidebar"] { background: #0f1116; }
                    [data-testid="stSidebar"] > div { background: #0f1116; }
                    @media (max-width: 640px) {
                        [data-testid="stHeader"],
                        [data-testid="stToolbar"] {
                            background-color: #0f1116 !important;
                            background: #0f1116 !important;
                        }
                        [data-testid="stSidebar"] {
                            background-color: #0f1116 !important;
                            background: #0f1116 !important;
                        }
                        [data-testid="stSidebar"] > div {
                            background-color: #0f1116 !important;
                            background: #0f1116 !important;
                        }
                    }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _build_crop_sensitivity_table() -> pd.DataFrame:
    """
    Return a crop sensitivity table with a stable Crop / Overall_Sensitivity shape.

    Prefer a dedicated crop sensitivity export when available, but fall back to the
    seasonal crop summary shipped with the results folder and derive a single overall
    sensitivity score from its seasonal columns.
    """

    dedicated = _read_csv(_RESULTS_DIR / "Crop_Sensitivity_Analysis.csv")
    if isinstance(dedicated, pd.DataFrame) and not dedicated.empty and {"Crop", "Overall_Sensitivity"}.issubset(dedicated.columns):
        return dedicated

    seasonal = _read_csv(_RESULTS_DIR / "seasonal_climate_sensitivity_by_crop.csv")
    if isinstance(seasonal, pd.DataFrame) and not seasonal.empty:
        seasonal = seasonal.copy()
        if "Crop" not in seasonal.columns:
            first_column = seasonal.columns[0]
            seasonal = seasonal.rename(columns={first_column: "Crop"})

        if "Crop" not in seasonal.columns:
            return pd.DataFrame()

        numeric_cols = [col for col in seasonal.columns if col != "Crop"]
        for col in numeric_cols:
            seasonal[col] = pd.to_numeric(seasonal[col], errors="coerce")

        seasonal["Overall_Sensitivity"] = seasonal[numeric_cols].abs().mean(axis=1, skipna=True)
        return seasonal[["Crop", "Overall_Sensitivity"]]

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, Any]:
    """
    Load precomputed artifacts produced by the notebook pipeline.

    Returns a dict keyed by what the Streamlit pages expect.
    Missing artifacts return empty DataFrames / empty values.
    """

    data: Dict[str, Any] = {}

    data["strategies"] = _read_csv(_RESULTS_DIR / "Adaptive_Strategies_Recommendations.csv")
    data["food_security"] = _read_csv(_RESULTS_DIR / "Food_Security_Assessment.csv")
    data["crop_sensitivity"] = _build_crop_sensitivity_table()

    # Some pages refer to this as "resilience" (index by crop-region).
    resilience_path = _RESULTS_DIR / "Resilience_Index_by_CropRegion.csv"
    data["resilience"] = _read_csv(resilience_path)

    # Ensemble metrics from TCN-MLP training
    data["ensemble_metrics"] = _read_csv(_RESULTS_DIR / "ensemble_metrics_mapes_maase_trimmed.csv")
    data["per_crop_ensemble"] = _read_csv(_RESULTS_DIR / "per_crop_ensemble_mapes_maase_trimmed.csv")
    data["region_ensemble_metrics"] = _read_csv(_RESULTS_DIR / "region_ensemble_metrics.csv")
    data["crop_region_resilience"] = _read_csv(_RESULTS_DIR / "crop_region_resilience.csv")
    data["processed_dataset"] = _read_csv(_ROOT / "data" / "processed_dataset.csv")

    data["metadata"] = _read_json(_MODELS_DIR / "TCN_MLP_Crops_Mini_metadata.json")
    data["summary"] = _read_text(_RESULTS_DIR / "TCN_MLP_4Crops_RESULTS.txt")

    return data

