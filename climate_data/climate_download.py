"""
NASA POWER Climate Data Downloader — Nigeria (All 37 States)
============================================================
Study   : Evaluating the Impact of Climate Change on Food Security in Nigeria
Data    : NASA POWER Daily Agroclimatology (AG community), MERRA-2
Period  : 1999-01-01 to 2023-12-31

Parameters by group
-------------------
  Temperature    : T2M, T2M_MAX, T2M_MIN, TS
  Rainfall       : PRECTOTCORR
  Humidity       : RH2M, QV2M, T2MDEW, T2MWET
  Solar          : ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, ALLSKY_SFC_LW_DWN, CLOUD_AMT
  Wind & Pressure: WS2M, PS
  Soil Moisture  : GWETTOP, GWETROOT

Column order in every output CSV
---------------------------------
  Date, Year, Month, Geopolitical_Zone, State, <parameter(s)>

Output structure
----------------
  data/raw/temperature.csv
  data/raw/rainfall.csv
  data/raw/humidity.csv
  data/raw/solar.csv
  data/raw/wind_pressure.csv
  data/raw/soil_moisture.csv
  data/combined/nasa_power_nigeria_all_1999_2023.csv
  logs/

Usage
-----
    pip install requests pandas tqdm
    python nasa_power_nigeria_download.py
"""

import os
import time
import logging
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

START_DATE = "19990101"
END_DATE   = "20231231"

# Parameter groups — each group saved as its own CSV in data/raw/
PARAMETER_GROUPS = {
    "temperature": [
        "T2M",        # Mean temperature at 2m (°C)
        "T2M_MAX",    # Max temperature at 2m (°C)
        "T2M_MIN",    # Min temperature at 2m (°C)
        "TS",         # Earth skin / land surface temperature (°C)
    ],
    "rainfall": [
        "PRECTOTCORR",  # Bias-corrected total precipitation (mm/day)
    ],
    "humidity": [
        "RH2M",    # Relative humidity at 2m (%)
        "QV2M",    # Specific humidity at 2m (g/kg)
        "T2MDEW",  # Dew point temperature at 2m (°C)
        "T2MWET",  # Wet bulb temperature at 2m (°C)
    ],
    "solar": [
        "ALLSKY_SFC_SW_DWN",  # All-sky surface shortwave downwelling (MJ/m²/day)
        "CLRSKY_SFC_SW_DWN",  # Clear-sky surface shortwave downwelling (MJ/m²/day)
        "ALLSKY_SFC_LW_DWN",  # All-sky surface longwave downwelling (W/m²)
        "CLOUD_AMT",          # Cloud amount (%)
    ],
    "wind_pressure": [
        "WS2M",  # Wind speed at 2m (m/s)
        "PS",    # Surface pressure (kPa)
    ],
    "soil_moisture": [
        "GWETTOP",   # Surface soil wetness — top layer (fraction, 0–1)
        "GWETROOT",  # Root zone soil wetness (fraction, 0–1)
    ],
}

# Flat list of all parameters for a single API call per state
ALL_PARAMETERS = [p for group in PARAMETER_GROUPS.values() for p in group]

# Nigeria — all 37 states (36 states + FCT)
NIGERIA_STATES = [
    {"Geopolitical_Zone": "North-Central", "State": "Benue",       "lat": 7.73,  "lon": 8.54},
    {"Geopolitical_Zone": "North-Central", "State": "FCT",         "lat": 9.07,  "lon": 7.48},
    {"Geopolitical_Zone": "North-Central", "State": "Kogi",        "lat": 7.80,  "lon": 6.74},
    {"Geopolitical_Zone": "North-Central", "State": "Kwara",       "lat": 8.48,  "lon": 4.54},
    {"Geopolitical_Zone": "North-Central", "State": "Nasarawa",    "lat": 8.49,  "lon": 7.71},
    {"Geopolitical_Zone": "North-Central", "State": "Niger",       "lat": 9.61,  "lon": 6.55},
    {"Geopolitical_Zone": "North-Central", "State": "Plateau",     "lat": 9.89,  "lon": 8.86},
    {"Geopolitical_Zone": "North-East",    "State": "Adamawa",     "lat": 9.21,  "lon": 12.48},
    {"Geopolitical_Zone": "North-East",    "State": "Bauchi",      "lat": 10.31, "lon": 9.84},
    {"Geopolitical_Zone": "North-East",    "State": "Borno",       "lat": 11.83, "lon": 13.15},
    {"Geopolitical_Zone": "North-East",    "State": "Gombe",       "lat": 10.28, "lon": 11.17},
    {"Geopolitical_Zone": "North-East",    "State": "Taraba",      "lat": 8.89,  "lon": 11.36},
    {"Geopolitical_Zone": "North-East",    "State": "Yobe",        "lat": 11.75, "lon": 11.96},
    {"Geopolitical_Zone": "North-West",    "State": "Jigawa",      "lat": 11.75, "lon": 9.34},
    {"Geopolitical_Zone": "North-West",    "State": "Kaduna",      "lat": 10.51, "lon": 7.44},
    {"Geopolitical_Zone": "North-West",    "State": "Kano",        "lat": 12.00, "lon": 8.52},
    {"Geopolitical_Zone": "North-West",    "State": "Katsina",     "lat": 12.99, "lon": 7.60},
    {"Geopolitical_Zone": "North-West",    "State": "Kebbi",       "lat": 12.45, "lon": 4.19},
    {"Geopolitical_Zone": "North-West",    "State": "Sokoto",      "lat": 13.06, "lon": 5.23},
    {"Geopolitical_Zone": "North-West",    "State": "Zamfara",     "lat": 12.16, "lon": 6.66},
    {"Geopolitical_Zone": "South-East",    "State": "Abia",        "lat": 5.52,  "lon": 7.49},
    {"Geopolitical_Zone": "South-East",    "State": "Anambra",     "lat": 6.21,  "lon": 7.07},
    {"Geopolitical_Zone": "South-East",    "State": "Ebonyi",      "lat": 6.32,  "lon": 8.11},
    {"Geopolitical_Zone": "South-East",    "State": "Enugu",       "lat": 6.44,  "lon": 7.50},
    {"Geopolitical_Zone": "South-East",    "State": "Imo",         "lat": 5.48,  "lon": 7.03},
    {"Geopolitical_Zone": "South-South",   "State": "Akwa Ibom",   "lat": 5.03,  "lon": 7.92},
    {"Geopolitical_Zone": "South-South",   "State": "Bayelsa",     "lat": 4.93,  "lon": 6.26},
    {"Geopolitical_Zone": "South-South",   "State": "Cross River", "lat": 4.95,  "lon": 8.33},
    {"Geopolitical_Zone": "South-South",   "State": "Delta",       "lat": 6.20,  "lon": 6.73},
    {"Geopolitical_Zone": "South-South",   "State": "Edo",         "lat": 6.34,  "lon": 5.62},
    {"Geopolitical_Zone": "South-South",   "State": "Rivers",      "lat": 4.81,  "lon": 7.01},
    {"Geopolitical_Zone": "South-West",    "State": "Ekiti",       "lat": 7.62,  "lon": 5.22},
    {"Geopolitical_Zone": "South-West",    "State": "Lagos",       "lat": 6.52,  "lon": 3.37},
    {"Geopolitical_Zone": "South-West",    "State": "Ogun",        "lat": 7.15,  "lon": 3.35},
    {"Geopolitical_Zone": "South-West",    "State": "Ondo",        "lat": 7.25,  "lon": 5.21},
    {"Geopolitical_Zone": "South-West",    "State": "Osun",        "lat": 7.77,  "lon": 4.56},
    {"Geopolitical_Zone": "South-West",    "State": "Oyo",         "lat": 7.38,  "lon": 3.93},
]

# Request settings
REQUEST_TIMEOUT  = 120  # seconds per API call
RETRY_LIMIT      = 3    # max retries on failure
RETRY_DELAY      = 10   # seconds between retries
RATE_LIMIT_DELAY = 2    # seconds between requests (polite to NASA API)

# Column order prefix for all output CSVs
ID_COLS = ["Date", "Year", "Month", "Geopolitical_Zone", "State"]

# Output directories
DIR_RAW      = Path("data/raw")
DIR_COMBINED = Path("data/combined")
DIR_CACHE    = Path("data/.cache")
DIR_LOGS     = Path("logs")


# ─── SETUP ─────────────────────────────────────────────────────────────────────

def setup_dirs():
    for d in [DIR_RAW, DIR_COMBINED, DIR_CACHE, DIR_LOGS]:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging():
    log_path = DIR_LOGS / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info("NASA POWER Nigeria Download — Started")
    logging.info(f"Period     : {START_DATE} → {END_DATE}")
    logging.info(f"States     : {len(NIGERIA_STATES)}")
    logging.info(f"Parameters : {len(ALL_PARAMETERS)} — {', '.join(ALL_PARAMETERS)}")
    logging.info(f"Groups     : {list(PARAMETER_GROUPS.keys())}")
    return log_path


# ─── API ───────────────────────────────────────────────────────────────────────

def build_api_params(lat: float, lon: float) -> dict:
    """Build NASA POWER API query — all groups fetched in one request per state."""
    return {
        "parameters": ",".join(ALL_PARAMETERS),
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      START_DATE,
        "end":        END_DATE,
        "format":     "JSON",
    }


def fetch_state(state_info: dict) -> pd.DataFrame | None:
    """
    Fetch all parameters for one state with retry logic.

    Returns a DataFrame with columns:
        Date, Year, Month, Geopolitical_Zone, State, <all 17 parameters>
    Returns None on failure.
    """
    state = state_info["State"]
    zone  = state_info["Geopolitical_Zone"]
    lat   = state_info["lat"]
    lon   = state_info["lon"]

    api_params = build_api_params(lat, lon)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            logging.info(f"[{state}] Attempt {attempt}/{RETRY_LIMIT} — lat={lat}, lon={lon}")

            response = requests.get(
                NASA_POWER_URL,
                params=api_params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            # NASA POWER response:
            # data["properties"]["parameter"][PARAM][YYYYMMDD] = value
            parameter_data = data.get("properties", {}).get("parameter", {})

            if not parameter_data:
                logging.warning(f"[{state}] Empty parameter data in response.")
                return None

            # Build DataFrame — rows are dates, columns are parameters
            df = pd.DataFrame(parameter_data)
            df.index.name = "Date"
            df.reset_index(inplace=True)

            # Parse date, extract Year and Month
            df["Date"]  = pd.to_datetime(df["Date"], format="%Y%m%d")
            df["Year"]  = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month

            # Add zone and state identifiers
            df["Geopolitical_Zone"] = zone
            df["State"]             = state

            # Replace NASA missing value sentinel (-999) with NaN
            df.replace(-999.0, float("nan"), inplace=True)
            df.replace(-999,   float("nan"), inplace=True)

            # Enforce column order: Date, Year, Month, Geopolitical_Zone, State, <params>
            param_cols = [p for p in ALL_PARAMETERS if p in df.columns]
            df = df[ID_COLS + param_cols]

            logging.info(
                f"[{state}] ✓  {len(df):,} rows | "
                f"{df['Date'].min().date()} → {df['Date'].max().date()} | "
                f"missing values: {df[param_cols].isnull().sum().sum()}"
            )
            return df

        except requests.exceptions.Timeout:
            logging.warning(f"[{state}] Timeout on attempt {attempt}.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"[{state}] HTTP {e.response.status_code}: {e}")
            if e.response.status_code == 429:
                logging.info(f"[{state}] Rate limited — waiting 60s ...")
                time.sleep(60)
        except requests.exceptions.RequestException as e:
            logging.error(f"[{state}] Request error: {e}")
        except (KeyError, ValueError) as e:
            logging.error(f"[{state}] Parse error: {e}")
            return None

        if attempt < RETRY_LIMIT:
            logging.info(f"[{state}] Retrying in {RETRY_DELAY}s ...")
            time.sleep(RETRY_DELAY)

    logging.error(f"[{state}] FAILED after {RETRY_LIMIT} attempts.")
    return None


# ─── SAVE ──────────────────────────────────────────────────────────────────────

def save_group_csvs(combined: pd.DataFrame):
    """
    Split the combined DataFrame by parameter group and save one CSV each.

    Output files:
        data/raw/temperature.csv   — Date,Year,Month,Geopolitical_Zone,State,T2M,T2M_MAX,T2M_MIN,TS
        data/raw/rainfall.csv      — Date,Year,Month,Geopolitical_Zone,State,PRECTOTCORR
        data/raw/humidity.csv      — Date,Year,Month,Geopolitical_Zone,State,RH2M,QV2M,T2MDEW,T2MWET
        data/raw/solar.csv         — Date,Year,Month,Geopolitical_Zone,State,ALLSKY_SFC_SW_DWN,...
        data/raw/wind_pressure.csv — Date,Year,Month,Geopolitical_Zone,State,WS2M,PS
        data/raw/soil_moisture.csv — Date,Year,Month,Geopolitical_Zone,State,GWETTOP,GWETROOT
    """
    for group_name, params in PARAMETER_GROUPS.items():
        available = [p for p in params if p in combined.columns]
        if not available:
            logging.warning(f"Group '{group_name}': no columns found in data, skipping.")
            continue

        group_df = combined[ID_COLS + available].copy()
        out_path = DIR_RAW / f"{group_name}.csv"
        group_df.to_csv(out_path, index=False)

        size_kb = os.path.getsize(out_path) / 1024
        logging.info(
            f"Saved data/raw/{group_name}.csv — "
            f"{len(group_df):,} rows | cols: {ID_COLS + available} | {size_kb:.0f} KB"
        )


def save_combined(combined: pd.DataFrame):
    """Save the full merged dataset (all groups, all states) to data/combined/."""
    out_path = DIR_COMBINED / "nasa_power_nigeria_all_1999_2023.csv"
    combined.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logging.info(
        f"\n{'='*60}\n"
        f"COMBINED FILE\n"
        f"  Path    : {out_path}\n"
        f"  Rows    : {len(combined):,}\n"
        f"  Size    : {size_mb:.1f} MB\n"
        f"  States  : {combined['State'].nunique()}\n"
        f"  Columns : {list(combined.columns)}\n"
        f"{'='*60}"
    )


def print_summary(results: dict):
    """Print final download and file summary."""
    success = [s for s, v in results.items() if v]
    failed  = [s for s, v in results.items() if not v]

    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total states : {len(results)}")
    print(f"  Successful   : {len(success)}")
    print(f"  Failed       : {len(failed)}")
    if failed:
        print(f"\n  FAILED STATES (re-run to retry):")
        for s in failed:
            print(f"    - {s}")

    print(f"\n  RAW GROUP FILES (data/raw/):")
    for group_name in PARAMETER_GROUPS:
        path = DIR_RAW / f"{group_name}.csv"
        if path.exists():
            size_kb = os.path.getsize(path) / 1024
            print(f"    {group_name}.csv  ({size_kb:,.0f} KB)")

    combined_path = DIR_COMBINED / "nasa_power_nigeria_all_1999_2023.csv"
    if combined_path.exists():
        size_mb = os.path.getsize(combined_path) / (1024 * 1024)
        print(f"\n  COMBINED FILE (data/combined/):")
        print(f"    nasa_power_nigeria_all_1999_2023.csv  ({size_mb:.1f} MB)")

    print(f"{'='*60}\n")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    setup_dirs()
    log_path = setup_logging()

    all_dfs = []
    results = {}

    print(f"\nNASA POWER Nigeria Climate Data Downloader")
    print(f"Period     : {START_DATE} → {END_DATE}")
    print(f"States     : {len(NIGERIA_STATES)}")
    print(f"Parameters : {len(ALL_PARAMETERS)} across {len(PARAMETER_GROUPS)} groups")
    print(f"Groups     : {', '.join(PARAMETER_GROUPS.keys())}")
    print(f"Est. time  : ~{len(NIGERIA_STATES) * (RATE_LIMIT_DELAY + 5) // 60} minutes\n")

    pbar = tqdm(NIGERIA_STATES, desc="Downloading", unit="state")

    for state_info in pbar:
        state = state_info["State"]
        pbar.set_description(f"{state:<20}")

        # Resume support: use per-state cache file
        cache_file = DIR_CACHE / f"{state.replace(' ', '_').lower()}.csv"

        if cache_file.exists():
            logging.info(f"[{state}] Cache hit — loading from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            all_dfs.append(df)
            results[state] = True
            continue

        df = fetch_state(state_info)

        if df is not None:
            df.to_csv(cache_file, index=False)   # cache for resume
            all_dfs.append(df)
            results[state] = True
        else:
            results[state] = False

        time.sleep(RATE_LIMIT_DELAY)

    # Build outputs from all collected DataFrames
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values(["Geopolitical_Zone", "State", "Date"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

        save_group_csvs(combined)
        save_combined(combined)

    print_summary(results)
    logging.info(f"Log → {log_path}")
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()