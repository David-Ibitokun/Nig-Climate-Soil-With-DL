# Deployment Guide

This document explains how to run the project locally, deploy the Streamlit dashboard, and an optional Docker-based deployment.

## Prerequisites

- Python 3.9+ (3.11/3.12/3.13 compatible)
- Git
- Recommended: 8+ GB disk space, 4+ GB RAM for model loading

## Local Deployment (Windows)

1. Clone the repository and change directory:

```powershell
git clone <repo-url>
cd project_test
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or .\.venv\Scripts\activate.bat
```

3. Install dependencies required for the Streamlit dashboard:

```powershell
pip install -r requirements-streamlit.txt
```

4. Start the Streamlit app:

```powershell
.\run_app.bat
# or: streamlit run app.py
```

Notes:
- If using `run_app.bat`, the script will activate the virtual environment and run Streamlit.
- If the app fails to load models due to memory constraints, consider reducing batch sizes or running on a machine with more RAM.

## Local Deployment (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-streamlit.txt
./run_app.sh
# or: streamlit run app.py
```

## Streamlit Cloud (quick)

1. Push repository to GitHub.
2. On Streamlit Cloud, create a new app and connect to the repository and branch.
3. Set the start command to `streamlit run app.py` and ensure `requirements-streamlit.txt` is present.
4. Provide any environment variables in the app settings (none are required by default).

Notes:
- Streamlit Cloud has memory limits; test model loading and consider lazy-loading or smaller models for cloud deployments.

## CI/CD & Automated Deployments (example)

- Use GitHub Actions to run tests and push Docker images to a registry. A minimal workflow will:
  - Install Python
  - Run unit checks (optional)
  - Build Docker image and push to Docker Hub/GHA packages

## Troubleshooting

- Model fails to load or OOM: increase available memory or load a smaller model variant.
- Missing packages: ensure `requirements-streamlit.txt` is up-to-date; consider `pip install -r requirements.txt` for full modeling stack.
- File path errors: confirm that `project_data/` and `New_Changes/results/` exist and are in the repository root.

## Recommended next steps after deployment

- Confirm that `New_Changes/results/` is writable (dashboard may read images/CSVs from this folder).
- Run the notebooks in `New_Changes/yield_changes/` to reproduce study results if needed.

---

Keep this guide in the repo root as `DEPLOYMENT_GUIDE.md`.
