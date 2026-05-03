# Climate Change & Food Security in Nigeria — TCN-MLP Project

This repository contains the code, notebooks, models, and results for a TCN-MLP hybrid analysis of climate impacts on crop yields and regional food security in Nigeria.

Status: active (core notebooks and Streamlit dashboard available)

## Quick Start (Windows)

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1    # or use .\\.venv\\Scripts\\activate.bat
```

2. Install dependencies (use the Streamlit-specific file for the dashboard):

```powershell
pip install -r requirements-streamlit.txt
```

3. Run the Streamlit app (recommended):

```powershell
.\\run_app.bat
# or: streamlit run app.py
```

4. Notebooks (optional): open the main analysis notebooks in `New_Changes/yield_changes/` with Jupyter Lab/Notebook.

## Kept Notebooks (main analysis)

- `New_Changes/yield_changes/01_Data_Preprocessing.ipynb`
- `New_Changes/yield_changes/02_Model_Training_Minimal.ipynb`
- `New_Changes/yield_changes/03_Model_Evaluation_Minimal.ipynb`
- `New_Changes/yield_changes/04_Climate_Impact.ipynb`
- `New_Changes/yield_changes/Combined.ipynb`

## Project layout (important locations)

- `app.py` — Streamlit dashboard entry
- `pages/` — Streamlit app pages
- `New_Changes/yield_changes/` — Primary notebooks and pipeline
- `New_Changes/results/` — Primary results and figures (canonical output location)
- `models/` and `New_Changes/models/` — Trained model files
- `project_data/` — Processed datasets used for training and analysis
- `climate_data/` — Climate download and raw climate files

## Running the analysis

- To reproduce model training and analysis, run the notebooks in `New_Changes/yield_changes/` in the order: 01 → 02 → 03 → 04. Use `Combined.ipynb` for an integrated pipeline.
- Save any generated plots or CSV outputs to `New_Changes/results/`.

## Dependencies

- See `requirements.txt` (modeling) and `requirements-streamlit.txt` (dashboard + lightweight deps).

## Notes on recent cleanup

- Old or duplicate notebooks were removed to keep the repository focused on the active pipeline (`New_Changes/yield_changes/`) and the Streamlit dashboard.

## Contact

- Author: David Ibitokun
- For questions: open an issue or email the project maintainer.

---

**Last updated**: May 2026
