"""
Complete Data Download Script for Climate-Food Security Project
Downloads ALL required climate data automatically from NASA POWER and OCO-2 APIs

Author: Climate-Food Security Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
from pathlib import Path
import sys
import warnings
import traceback
import json
import os
from dotenv import load_dotenv

try:
    import h5py
except ImportError:
    h5py = None
    print("WARNING: h5py not installed. Run: pip install h5py")
    print("         h5py is required to read OCO-2 HDF5 files.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Load Earthdata credentials from .env
load_dotenv(Path("project_data/raw_data/climate/.env"))

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings"""
    
    # Directories
    BASE_DIR = Path("project_data")
    RAW_DATA_DIR = BASE_DIR / "raw_data"
    CLIMATE_DIR = RAW_DATA_DIR / "climate"
    AGRICULTURE_DIR = RAW_DATA_DIR / "agriculture"
    SOIL_DIR = RAW_DATA_DIR / "soil"
    
    # Time period
    START_YEAR = 2000
    END_YEAR = 2023
    
    # NASA POWER API - CORRECTED ENDPOINT for daily data
    NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    # NASA OCO-2 (Earthdata GES DISC) for state-specific XCO2
    # Credentials: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in
    #              project_data/raw_data/climate/.env
    EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME", "")
    EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD", "")

    # AIRS CMR short name (mid-tropospheric CO2, 2002-present)
    AIRS_SHORT_NAME = "AIRS3STM"   # AIRS/Aqua L3 Monthly Standard Physical Retrieval
    AIRS_VERSION    = "7.0"

    # OCO-2 CMR granule search endpoint
    CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
    # OCO-2 L2 Lite FP v10r short name (covers Sep 2014-present; use 2015 for full years)
    OCO2_SHORT_NAME = "OCO2_L2_Lite_FP"
    OCO2_VERSION   = "10r"       # Use "09r" if 10r is unavailable
    OCO2_START     = 2015        # Use 2015: OCO-2 launched Jul 2014, full-year data from 2015
    AIRS_START     = 2002        # NASA AIRS/Aqua satellite data availability (covers 2002-2014)
    # Nigeria bounding box for granule search (W,S,E,N)
    NIGERIA_BBOX   = "3.0,4.0,15.0,14.0"
    # Radius (degrees) used to match OCO-2 soundings to a state capital
    OCO2_RADIUS_DEG = 0.75
    
    # Nigerian geopolitical zones with all states and coordinates
    ZONES = {
        'North-Central': {
            'Benue': {'lat': 7.73, 'lon': 8.54},
            'FCT': {'lat': 9.07, 'lon': 7.48},
            'Kogi': {'lat': 7.80, 'lon': 6.74},
            'Kwara': {'lat': 8.48, 'lon': 4.54},
            'Nasarawa': {'lat': 8.49, 'lon': 7.71},
            'Niger': {'lat': 9.61, 'lon': 6.55},
            'Plateau': {'lat': 9.89, 'lon': 8.86}
        },
        'North-East': {
            'Adamawa': {'lat': 9.21, 'lon': 12.48},
            'Bauchi': {'lat': 10.31, 'lon': 9.84},
            'Borno': {'lat': 11.83, 'lon': 13.15},
            'Gombe': {'lat': 10.28, 'lon': 11.17},
            'Taraba': {'lat': 8.89, 'lon': 11.36},
            'Yobe': {'lat': 11.75, 'lon': 11.96}
        },
        'North-West': {
            'Jigawa': {'lat': 11.75, 'lon': 9.34},
            'Kaduna': {'lat': 10.51, 'lon': 7.44},
            'Kano': {'lat': 12.00, 'lon': 8.52},
            'Katsina': {'lat': 12.99, 'lon': 7.60},
            'Kebbi': {'lat': 12.45, 'lon': 4.19},
            'Sokoto': {'lat': 13.06, 'lon': 5.23},
            'Zamfara': {'lat': 12.16, 'lon': 6.66}
        },
        'South-East': {
            'Abia': {'lat': 5.52, 'lon': 7.49},
            'Anambra': {'lat': 6.21, 'lon': 7.07},
            'Ebonyi': {'lat': 6.32, 'lon': 8.11},
            'Enugu': {'lat': 6.44, 'lon': 7.50},
            'Imo': {'lat': 5.48, 'lon': 7.03}
        },
        'South-South': {
            'Akwa Ibom': {'lat': 5.03, 'lon': 7.92},
            'Bayelsa': {'lat': 4.93, 'lon': 6.26},
            'Cross River': {'lat': 4.95, 'lon': 8.33},
            'Delta': {'lat': 6.20, 'lon': 6.73},
            'Edo': {'lat': 6.34, 'lon': 5.62},
            'Rivers': {'lat': 4.81, 'lon': 7.01}
        },
        'South-West': {
            'Ekiti': {'lat': 7.62, 'lon': 5.22},
            'Lagos': {'lat': 6.52, 'lon': 3.37},
            'Ogun': {'lat': 7.15, 'lon': 3.35},
            'Ondo': {'lat': 7.25, 'lon': 5.21},
            'Osun': {'lat': 7.77, 'lon': 4.56},
            'Oyo': {'lat': 7.38, 'lon': 3.93}
        }
    }


"""
use this instead
{"Geopolitical_Zone": "North-Central", "State": "Benue", "lat": 7.73, "lon": 8.54},
    {"Geopolitical_Zone": "North-Central", "State": "FCT", "lat": 9.07, "lon": 7.48},
    {"Geopolitical_Zone": "North-Central", "State": "Kogi", "lat": 7.80, "lon": 6.74},
    {"Geopolitical_Zone": "North-Central", "State": "Kwara", "lat": 8.48, "lon": 4.54},
    {"Geopolitical_Zone": "North-Central", "State": "Nasarawa", "lat": 8.49, "lon": 7.71},
    {"Geopolitical_Zone": "North-Central", "State": "Niger", "lat": 9.61, "lon": 6.55},
    {"Geopolitical_Zone": "North-Central", "State": "Plateau", "lat": 9.89, "lon": 8.86},
    {"Geopolitical_Zone": "North-East", "State": "Adamawa", "lat": 9.21, "lon": 12.48},
    {"Geopolitical_Zone": "North-East", "State": "Bauchi", "lat": 10.31, "lon": 9.84},
    {"Geopolitical_Zone": "North-East", "State": "Borno", "lat": 11.83, "lon": 13.15},
    {"Geopolitical_Zone": "North-East", "State": "Gombe", "lat": 10.28, "lon": 11.17},
    {"Geopolitical_Zone": "North-East", "State": "Taraba", "lat": 8.89, "lon": 11.36},
    {"Geopolitical_Zone": "North-East", "State": "Yobe", "lat": 11.75, "lon": 11.96},
    {"Geopolitical_Zone": "North-West", "State": "Jigawa", "lat": 11.75, "lon": 9.34},
    {"Geopolitical_Zone": "North-West", "State": "Kaduna", "lat": 10.51, "lon": 7.44},
    {"Geopolitical_Zone": "North-West", "State": "Kano", "lat": 12.00, "lon": 8.52},
    {"Geopolitical_Zone": "North-West", "State": "Katsina", "lat": 12.99, "lon": 7.60},
    {"Geopolitical_Zone": "North-West", "State": "Kebbi", "lat": 12.45, "lon": 4.19},
    {"Geopolitical_Zone": "North-West", "State": "Sokoto", "lat": 13.06, "lon": 5.23},
    {"Geopolitical_Zone": "North-West", "State": "Zamfara", "lat": 12.16, "lon": 6.66},
    {"Geopolitical_Zone": "South-East", "State": "Abia", "lat": 5.52, "lon": 7.49},
    {"Geopolitical_Zone": "South-East", "State": "Anambra", "lat": 6.21, "lon": 7.07},
    {"Geopolitical_Zone": "South-East", "State": "Ebonyi", "lat": 6.32, "lon": 8.11},
    {"Geopolitical_Zone": "South-East", "State": "Enugu", "lat": 6.44, "lon": 7.50},
    {"Geopolitical_Zone": "South-East", "State": "Imo", "lat": 5.48, "lon": 7.03},
    {"Geopolitical_Zone": "South-South", "State": "Akwa Ibom", "lat": 5.03, "lon": 7.92},
    {"Geopolitical_Zone": "South-South", "State": "Bayelsa", "lat": 4.93, "lon": 6.26},
    {"Geopolitical_Zone": "South-South", "State": "Cross River", "lat": 4.95, "lon": 8.33},
    {"Geopolitical_Zone": "South-South", "State": "Delta", "lat": 6.20, "lon": 6.73},
    {"Geopolitical_Zone": "South-South", "State": "Edo", "lat": 6.34, "lon": 5.62},
    {"Geopolitical_Zone": "South-South", "State": "Rivers", "lat": 4.81, "lon": 7.01},
    {"Geopolitical_Zone": "South-West", "State": "Ekiti", "lat": 7.62, "lon": 5.22},
    {"Geopolitical_Zone": "South-West", "State": "Lagos", "lat": 6.52, "lon": 3.37},
    {"Geopolitical_Zone": "South-West", "State": "Ogun", "lat": 7.15, "lon": 3.35},
    {"Geopolitical_Zone": "South-West", "State": "Ondo", "lat": 7.25, "lon": 5.21},
    {"Geopolitical_Zone": "South-West", "State": "Osun", "lat": 7.77, "lon": 4.56},
    {"Geopolitical_Zone": "South-West", "State": "Oyo", "lat": 7.38, "lon": 3.93}
"""


def create_directories():
    """Create project directory structure"""
    dirs = [Config.CLIMATE_DIR, Config.AGRICULTURE_DIR, Config.SOIL_DIR]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    print("Directory structure created")

def print_header():
    """Print script header"""
    print("\n" + "="*70)
    print("CLIMATE-FOOD SECURITY DATA DOWNLOAD SCRIPT")
    print("="*70)
    print(f"Period: {Config.START_YEAR}-{Config.END_YEAR}")
    total_states = sum(len(states) for states in Config.ZONES.values())
    print(f"Coverage: {len(Config.ZONES)} geopolitical zones, {total_states} states")
    print("\nData Sources:")
    print("  1. CO2 (2015+): NASA OCO-2 L2 Lite FP (per-state XCO2 from satellite)")
    print("  2. Temperature : NASA POWER API")
    print("  3. Rainfall    : NASA POWER API")
    print("  4. Humidity    : NASA POWER API")
    print(f"\nNote: Comprehensive coverage of all 36 Nigerian states + FCT")
    print("Note: Crop yield data must be downloaded manually from FAO")
    print("="*70 + "\n")

# ============================================================================
# 1. NASA OCO-2 CO2 DATA (2015-present)
# ============================================================================
# ============================================================================

class _EarthdataSession(requests.Session):
    """
    NASA-recommended session class for Earthdata downloads (GES DISC pattern).

    Sends Authorization header ONLY when redirected to urs.earthdata.nasa.gov.
    Without this, basic-auth credentials are forwarded to every redirect hop,
    causing 401 at the OAuth authorize endpoint.

    Reference: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    """
    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            original_host = requests.utils.urlparse(response.request.url).hostname
            redirect_host = requests.utils.urlparse(url).hostname
            # Drop auth header unless we are talking to the URS auth host
            if (original_host != redirect_host
                    and redirect_host != self.AUTH_HOST
                    and original_host != self.AUTH_HOST):
                del headers["Authorization"]
        return


def _oco2_session():
    """Return an authenticated requests.Session for NASA Earthdata."""
    if not Config.EARTHDATA_USERNAME or not Config.EARTHDATA_PASSWORD:
        raise RuntimeError(
            "Missing Earthdata credentials.\n"
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in\n"
            "project_data/raw_data/climate/.env"
        )
    session = _EarthdataSession(Config.EARTHDATA_USERNAME, Config.EARTHDATA_PASSWORD)
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)
    return session


def _search_oco2_granules(year, month, session):
    """
    Use the CMR API to find OCO-2 Lite FP granules that intersect Nigeria
    for the given year/month.
    Returns a list of download URLs (HTTPS links to HDF5 files).
    """
    start_dt = f"{year}-{month:02d}-01T00:00:00Z"
    # Last day of month
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    end_dt   = f"{year}-{month:02d}-{last_day:02d}T23:59:59Z"

    params = {
        "short_name":        Config.OCO2_SHORT_NAME,
        "version":           Config.OCO2_VERSION,
        "temporal":          f"{start_dt},{end_dt}",
        "bounding_box":      Config.NIGERIA_BBOX,
        "page_size":         50,
        "sort_key":          "-start_date",
    }
    try:
        # CMR search is public — do NOT use authenticated session
        resp = requests.get(Config.CMR_URL, params=params, timeout=30)
        resp.raise_for_status()
        entries = resp.json().get("feed", {}).get("entry", [])
        urls = []
        for entry in entries:
            for link in entry.get("links", []):
                href = link.get("href", "")
                if href.endswith(".h5") or href.endswith(".nc4"):
                    urls.append(href)
                    break
        return urls
    except Exception as e:
        print(f"      CMR search error ({year}-{month:02d}): {e}")
        return []


def _extract_xco2_for_states(h5_path, zones):
    """
    Open an OCO-2 HDF5 Lite FP file and extract mean XCO2 (ppm) for each
    state capital within Config.OCO2_RADIUS_DEG degrees.

    Returns dict: {(zone, state): xco2_mean}
    """
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    results = {}
    try:
        with h5py.File(h5_path, "r") as hf:
            # Standard OCO-2 Lite FP variable paths
            lat_arr  = hf["latitude"][:]           # (n,)
            lon_arr  = hf["longitude"][:]          # (n,)
            xco2_arr = hf["xco2"][:]               # (n,)  units: ppm
            qf_arr   = hf["xco2_quality_flag"][:]  # 0 = good

            # Filter: quality flag == 0 only
            mask = qf_arr == 0
            lat_g  = lat_arr[mask]
            lon_g  = lon_arr[mask]
            xco2_g = xco2_arr[mask]

            for zone, states in zones.items():
                for state, coords in states.items():
                    dlat = lat_g - coords["lat"]
                    dlon = lon_g - coords["lon"]
                    dist = np.sqrt(dlat**2 + dlon**2)
                    nearby = dist < Config.OCO2_RADIUS_DEG
                    if nearby.sum() > 0:
                        results[(zone, state)] = float(np.nanmean(xco2_g[nearby]))
    except Exception as e:
        print(f"      HDF5 parse error: {e}")
    return results


def download_oco2_co2_data(session):
    """
    Download NASA OCO-2 XCO2 for all Nigerian states, 2014–END_YEAR.
    For each month:
      1. Search CMR for granules over Nigeria.
      2. Download each HDF5 file.
      3. Extract per-state XCO2 means.
      4. Save monthly state averages.
    Returns a DataFrame with columns:
        Year, Month, Geopolitical_Zone, State, CO2_ppm, Source
    """
    oco2_dir = Config.CLIMATE_DIR / "oco2_cache"
    oco2_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_months = (Config.END_YEAR - Config.OCO2_START + 1) * 12
    done = 0

    for year in range(Config.OCO2_START, Config.END_YEAR + 1):
        for month in range(1, 13):
            done += 1
            print(f"   OCO-2 [{done}/{total_months}] {year}-{month:02d} ...", end=" ", flush=True)

            # Monthly state accumulators
            state_xco2 = {}          # (zone, state) -> list of xco2 values

            granule_urls = _search_oco2_granules(year, month, session)
            if not granule_urls:
                print("no granules found — marking for interpolation")
                # No OCO-2 data available for this month; will be filled by interpolation
                for zone, states in Config.ZONES.items():
                    for state in states:
                        state_xco2.setdefault((zone, state), [])  # empty → filled below
            else:
                for url in granule_urls:
                    fname = oco2_dir / Path(url).name
                    # Download only if not cached (or cached file is empty/corrupt)
                    if fname.exists() and fname.stat().st_size < 1024:
                        fname.unlink()  # delete suspiciously small cached file
                    if not fname.exists():
                        try:
                            r = session.get(url, timeout=120, stream=True)
                            r.raise_for_status()
                            with open(fname, "wb") as fh:
                                for chunk in r.iter_content(chunk_size=1 << 20):
                                    fh.write(chunk)
                        except Exception as e:
                            print(f"download error ({Path(url).name}): {e}")
                            if fname.exists():
                                fname.unlink()  # remove partial download
                            continue

                    # Extract XCO2 per state
                    extracted = _extract_xco2_for_states(fname, Config.ZONES)
                    for key, val in extracted.items():
                        state_xco2.setdefault(key, []).append(val)

            # Build records for this month
            for zone, states in Config.ZONES.items():
                for state in states:
                    vals = state_xco2.get((zone, state), [])
                    co2_val = float(np.nanmean(vals)) if vals else np.nan
                    records.append({
                        "Year":              year,
                        "Month":             month,
                        "Geopolitical_Zone": zone,
                        "State":             state,
                        "CO2_ppm":           round(co2_val, 3),
                        "Source":            "OCO-2" if vals else "missing",
                    })

            ok_states = sum(1 for v in state_xco2.values() if v)
            print(f"{ok_states}/{sum(len(s) for s in Config.ZONES.values())} states covered")

    df = pd.DataFrame(records)
    return df


# ============================================================================
# 1. MAIN CO2 DOWNLOAD — OCO-2 (2015-present)
# ============================================================================

def download_co2_data():
    """
    Download state-specific CO2 data from NASA OCO-2 L2 Lite FP satellite data.
    
    Coverage: 2015 to present (OCO-2 launched in 2014, full-year data from 2015)

    Credentials required (in project_data/raw_data/climate/.env):
        EARTHDATA_USERNAME=your_username
        EARTHDATA_PASSWORD=your_password
    Register free at https://urs.earthdata.nasa.gov/
    """
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING CO2 DATA (NASA OCO-2)")
    print("="*70)
    print("Source:")
    print(f"  • {Config.OCO2_START}–{Config.END_YEAR}: NASA OCO-2 L2 Lite FP (per-state XCO2)")
    print("  • Dataset: https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_FP/")

    all_records = []

    # ── OCO-2 satellite data ─────────────────────────────────────────────────
    print(f"\nNASA OCO-2 satellite data ({Config.OCO2_START}–{Config.END_YEAR})...")

    if h5py is None:
        print("   ✗ h5py not installed — skipping OCO-2 download")
        print("     Run: pip install h5py  then re-run this script.")
        oco2_df = pd.DataFrame()
    elif not Config.EARTHDATA_USERNAME:
        print("   ✗ Earthdata credentials not set — skipping OCO-2 download")
        print("     Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD in")
        print("     project_data/raw_data/climate/.env")
        oco2_df = pd.DataFrame()
    else:
        try:
            session = _oco2_session()
            print(f"   Authenticated as: {Config.EARTHDATA_USERNAME}")
            oco2_df = download_oco2_co2_data(session)
            print(f"   ✓ OCO-2 records: {len(oco2_df):,}")
        except RuntimeError as e:
            print(f"   ✗ {e}")
            oco2_df = pd.DataFrame()
        except Exception as e:
            print(f"   ✗ OCO-2 download failed: {e}")
            oco2_df = pd.DataFrame()

    if not oco2_df.empty:
        all_records.extend(oco2_df.to_dict("records"))

    if not all_records:
        print("\nERROR: No CO2 data collected from any source.")
        return None

    df = pd.DataFrame(all_records).sort_values(["Year", "Month", "State"]).reset_index(drop=True)

    # Interpolate any remaining missing OCO-2 values using linear time interpolation
    # per state group
    if "missing" in df["Source"].values:
        print("   Interpolating missing OCO-2 months from nearest observations...")
        df["CO2_ppm"] = (
            df.sort_values(["State", "Year", "Month"])
              .groupby("State")["CO2_ppm"]
              .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )
        df.loc[df["Source"] == "missing", "Source"] = "interpolated"

    # CO2 growth rate per state
    df = df.sort_values(["State", "Year", "Month"])
    df["CO2_Growth_Rate_ppm_per_year"] = (
        df.groupby(["State", "Month"])["CO2_ppm"].diff(12).round(4)
    )

    output_file = Config.CLIMATE_DIR / "co2_data.csv"
    df.to_csv(output_file, index=False)

    print(f"\nCO2 DATA SAVED SUCCESSFULLY")
    print(f"   Total records : {len(df):,}")
    print(f"   Period        : {df['Year'].min()}–{df['Year'].max()}")
    print(f"   States        : {df['State'].nunique()}")
    print(f"   CO2 range     : {df['CO2_ppm'].min():.2f} – {df['CO2_ppm'].max():.2f} ppm")
    src_counts = df['Source'].value_counts().to_dict()
    print(f"   Sources       : {src_counts}")
    print(f"   File          : {output_file}")

    return df

# ============================================================================
# 2. NASA POWER DATA (Temperature, Rainfall, Humidity) - CORRECTED
# ============================================================================

def download_nasa_power_data(lat, lon, start_year, end_year, state_name):
    """
    Download climate data from NASA POWER API using DAILY endpoint
    Then aggregate to monthly data
    
    Daily Parameters:
    - T2M: Temperature at 2 Meters (°C)
    - T2M_MAX: Maximum Temperature (°C)
    - T2M_MIN: Minimum Temperature (°C)
    - PRECTOTCORR: Precipitation Corrected (mm/day)
    - RH2M: Relative Humidity (%)
    """
    
    # Parameters for daily data - using correct parameter names
    parameters = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M']
    
    def try_request(start_date, end_date):
        """Helper function to try a single request"""
        params = {
            'parameters': ','.join(parameters),
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': start_date,
            'end': end_date,
            'format': 'JSON'
        }
        
        try:
            response = requests.get(Config.NASA_POWER_URL, params=params, timeout=180)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Try to get error details
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        print(f"    API Error: {error_data['message'][:100]}")
                    else:
                        print(f"    HTTP {response.status_code}")
                except:
                    print(f"    HTTP {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"    Timeout for {state_name}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"    Request error: {str(e)}")
            return None
        except Exception as e:
            print(f"    Unexpected error: {str(e)}")
            return None

    # Try 2-year chunks to avoid timeouts
    print(f"    Downloading {state_name} ({lat}, {lon})...")
    
    all_data = {
        'dates': [],
        'temps_avg': [],
        'temps_max': [],
        'temps_min': [],
        'rainfall': [],
        'humidity': []
    }
    
    chunk_size = 2  # 2-year chunks to be safe
    all_chunks_successful = True
    
    for year_start in range(start_year, end_year + 1, chunk_size):
        year_end = min(year_start + chunk_size - 1, end_year)
        
        start_date = f"{year_start}0101"
        end_date = f"{year_end}1231"
        
        print(f"    Chunk {year_start}-{year_end}...", end=" ")
        
        data = None
        for attempt in range(3):  # Try up to 3 times per chunk
            data = try_request(start_date, end_date)
            if data is not None:
                break
            elif attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"Retry {attempt + 1} in {wait_time}s...", end=" ")
                time.sleep(wait_time)
        
        if data is None:
            print("FAILED")
            all_chunks_successful = False
            continue
        
        # Process the chunk data
        try:
            properties = data.get('properties', {}).get('parameter', {})
            
            # Extract daily data
            if 'T2M' in properties:
                for date_str, temp_avg in properties['T2M'].items():
                    try:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        
                        # Create date object
                        date_obj = datetime(year, month, day)
                        date_key = f"{year}-{month:02d}-01"  # Monthly aggregation key
                        
                        if date_key not in all_data['dates']:
                            all_data['dates'].append(date_key)
                            all_data['temps_avg'].append([])
                            all_data['temps_max'].append([])
                            all_data['temps_min'].append([])
                            all_data['rainfall'].append([])
                            all_data['humidity'].append([])
                        
                        idx = all_data['dates'].index(date_key)
                        
                        # Add daily values
                        all_data['temps_avg'][idx].append(temp_avg)
                        all_data['temps_max'][idx].append(properties['T2M_MAX'][date_str])
                        all_data['temps_min'][idx].append(properties['T2M_MIN'][date_str])
                        all_data['rainfall'][idx].append(properties['PRECTOTCORR'][date_str])
                        all_data['humidity'][idx].append(properties['RH2M'][date_str])
                        
                    except (KeyError, ValueError, IndexError) as e:
                        continue
            
            print("DONE")
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            all_chunks_successful = False
            continue
        
        # Be nice to the API
        time.sleep(1)
    
    if not all_data['dates']:
        print(f"    No data downloaded for {state_name}")
        return None
    
    # Aggregate daily data to monthly with proper calculation of daily extremes
    print(f"    Aggregating to monthly data...", end=" ")
    
    monthly_data = []
    
    for i, month_key in enumerate(all_data['dates']):
        if (all_data['temps_avg'][i] and all_data['temps_max'][i] and 
            all_data['temps_min'][i] and all_data['rainfall'][i] and 
            all_data['humidity'][i]):
            
            # Calculate monthly statistics from daily values
            temps_avg_list = all_data['temps_avg'][i]
            temps_max_list = all_data['temps_max'][i]
            temps_min_list = all_data['temps_min'][i]
            rainfall_list = all_data['rainfall'][i]
            humidity_list = all_data['humidity'][i]
            
            # Temperature statistics
            avg_temp = np.mean(temps_avg_list)
            max_temp = np.max(temps_max_list)
            min_temp = np.min(temps_min_list)
            
            # Rainfall statistics
            total_rainfall = np.sum(rainfall_list)
            max_daily_rainfall = np.max(rainfall_list)  # Actual max daily rainfall
            rainy_days = sum(1 for r in rainfall_list if r > 1.0)  # Days with >1mm rainfall
            
            # Humidity statistics
            avg_humidity = np.mean(humidity_list)
            max_humidity = np.max(humidity_list)
            min_humidity = np.min(humidity_list)
            
            # Count heat stress days (daily max > 35°C)
            heat_stress_days = sum(1 for t in temps_max_list if t > 35)
            
            # Count cold stress days (daily min < 10°C)
            cold_stress_days = sum(1 for t in temps_min_list if t < 10)
            
            monthly_data.append({
                'Date': month_key,
                'Avg_Temp_C': round(avg_temp, 2),
                'Max_Temp_C': round(max_temp, 2),
                'Min_Temp_C': round(min_temp, 2),
                'Rainfall_mm': round(total_rainfall, 1),
                'Max_Daily_Rainfall_mm': round(max_daily_rainfall, 1),
                'Rainy_Days': rainy_days,
                'Heat_Stress_Days': heat_stress_days,
                'Cold_Stress_Days': cold_stress_days,
                'Avg_Humidity_Percent': round(avg_humidity, 1),
                'Max_Humidity_Percent': round(max_humidity, 1),
                'Min_Humidity_Percent': round(min_humidity, 1)
            })
    
    df = pd.DataFrame(monthly_data)
    
    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    print("DONE")
    
    if len(df) > 0:
        print(f"    Downloaded {len(df)} months of data for {state_name}")
        return df
    else:
        print(f"    No valid monthly data for {state_name}")
        return None

def collect_all_nasa_climate_data():
    """Collect climate data from NASA POWER for all states"""
    print("\n" + "="*70)
    print("STEP 2-4: DOWNLOADING CLIMATE DATA FROM NASA POWER")
    print("="*70)
    print("Source: NASA Langley Research Center")
    print("API: https://power.larc.nasa.gov/")
    print("\nNOTE: Using DAILY data aggregated to MONTHLY")
    print("This will take approximately 20-30 minutes...")
    print("Please be patient - downloading data for 37 states\n")
    print("="*70 + "\n")
    
    all_temperature_data = []
    all_rainfall_data = []
    all_humidity_data = []
    
    total_states = sum(len(states) for states in Config.ZONES.values())
    current_state = 0
    failed_states = []
    
    start_time = time.time()
    
    for zone, states in Config.ZONES.items():
        print(f"\n{zone}:")
        print("-" * 70)
        
        for state, coords in states.items():
            current_state += 1
            
            print(f"[{current_state}/{total_states}] {state:15s} ", end="")
            sys.stdout.flush()
            
            df = download_nasa_power_data(
                coords['lat'],
                coords['lon'],
                Config.START_YEAR,
                Config.END_YEAR,
                state
            )
            
            if df is not None and len(df) > 0:
                # Add metadata
                df['Geopolitical_Zone'] = zone
                df['State'] = state
                df['Year'] = pd.to_datetime(df['Date']).dt.year
                df['Month'] = pd.to_datetime(df['Date']).dt.month
                
                # Process Temperature data
                temp_df = df[['Date', 'Year', 'Month', 'Geopolitical_Zone', 'State', 
                             'Avg_Temp_C', 'Min_Temp_C', 'Max_Temp_C', 
                             'Heat_Stress_Days', 'Cold_Stress_Days']].copy()
                temp_df['Temp_Range_C'] = (temp_df['Max_Temp_C'] - temp_df['Min_Temp_C']).round(1)
                all_temperature_data.append(temp_df)
                
                # Process Rainfall data
                rain_df = df[['Date', 'Year', 'Month', 'Geopolitical_Zone', 'State', 
                             'Rainfall_mm', 'Max_Daily_Rainfall_mm', 'Rainy_Days']].copy()
                rain_df['Rainfall_mm'] = rain_df['Rainfall_mm'].round(1)
                rain_df['Rainfall_Intensity'] = (rain_df['Rainfall_mm'] / 
                                                  rain_df['Rainy_Days'].replace(0, 1)).round(2)
                
                # Calculate precipitation anomaly index (relative to monthly average)
                expected_rainfall = rain_df.groupby(['State', 'Month'])['Rainfall_mm'].transform('mean')
                rain_df['Precipitation_Anomaly_Index'] = (1 - (rain_df['Rainfall_mm'] / expected_rainfall.replace(0, 1))).clip(0, 1).round(3)
                
                # Calculate flood risk index (combines intensity and anomaly)
                rain_df['Flood_Risk_Index'] = (
                    (rain_df['Max_Daily_Rainfall_mm'] / rain_df['Rainfall_mm'].replace(0, 1)) * 
                    (rain_df['Rainfall_mm'] / expected_rainfall.replace(0, 1))
                ).clip(0, 1).round(3)
                
                all_rainfall_data.append(rain_df)
                
                # Process Humidity data - using actual min/max from daily data
                humid_df = df[['Date', 'Year', 'Month', 'Geopolitical_Zone', 'State', 
                              'Avg_Humidity_Percent', 'Max_Humidity_Percent', 'Min_Humidity_Percent']].copy()
                humid_df['Humidity_Range_Percent'] = (humid_df['Max_Humidity_Percent'] - 
                                                       humid_df['Min_Humidity_Percent']).round(1)
                all_humidity_data.append(humid_df)
                
                print("DONE")
            else:
                print("FAILED")
                failed_states.append(state)
            
            # Rate limiting
            time.sleep(3)
    
    elapsed_time = time.time() - start_time
    
    # Combine and save data
    if all_temperature_data:
        print("\n" + "="*70)
        print("SAVING CLIMATE DATA FILES")
        print("="*70)
        
        # Temperature
        print("\n1. Temperature data...", end=" ")
        temp_full = pd.concat(all_temperature_data, ignore_index=True)
        temp_file = Config.CLIMATE_DIR / "temperature_data.csv"
        temp_full.to_csv(temp_file, index=False)
        print(f"Saved ({len(temp_full):,} records)")
        
        # Rainfall
        print("2. Rainfall data...", end=" ")
        rain_full = pd.concat(all_rainfall_data, ignore_index=True)
        rain_file = Config.CLIMATE_DIR / "rainfall_data.csv"
        rain_full.to_csv(rain_file, index=False)
        print(f"Saved ({len(rain_full):,} records)")
        
        # Humidity
        print("3. Humidity data...", end=" ")
        humid_full = pd.concat(all_humidity_data, ignore_index=True)
        humid_file = Config.CLIMATE_DIR / "humidity_data.csv"
        humid_full.to_csv(humid_file, index=False)
        print(f"Saved ({len(humid_full):,} records)")
        
        print(f"\nCLIMATE DATA DOWNLOADED SUCCESSFULLY")
        print(f"   Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Successful states: {total_states - len(failed_states)}/{total_states}")
        
        if failed_states:
            print(f"\nWARNING: Failed to download data for: {', '.join(failed_states)}")
            print("   You may need to retry for these states")
        
        return temp_full, rain_full, humid_full
    else:
        print("\nERROR: No climate data was successfully downloaded")
        print("   Please check your internet connection and NASA POWER API status")
        return None, None, None

# ============================================================================
# 3. CREATE DATA DOWNLOAD INSTRUCTIONS
# ============================================================================

def create_fao_download_instructions():
    """Create instructions for manual FAO download"""
    print("\n" + "="*70)
    print("STEP 5: CROP YIELD DATA (MANUAL DOWNLOAD REQUIRED)")
    print("="*70)
    
    instructions = """
INSTRUCTIONS FOR DOWNLOADING FAO CROP YIELD DATA:

1. Visit FAO Statistics:
    URL: https://www.fao.org/faostat/en/#data/QCL

2. Select the following:
    - Country: Nigeria
    - Element: Choose ALL of these:
      - Area harvested
      - Production
      - Yield
    - Item (Crops): Select the following crops:
      - Maize (corn)
      - Cassava
      - Yam
      - Rice
      - Sorghum
      - Millet
    - Years: 1990 to 2023
   
3. Click "Download" button (top right)
    - Format: CSV
    - Download the file

4. Save the downloaded file as:
    project_data/raw_data/agriculture/fao_crop_yield_raw.csv

5. After saving, run the preprocessing script to process the FAO data

Expected file columns:
Domain Code, Domain, Area Code (FAO), Area, Element Code, Element, 
Item Code (FAO), Item, Year Code, Year, Unit, Value, Flag

IMPORTANT: Make sure to download data for ALL years (1990-2023)
              and ALL crops listed above!

TROUBLESHOOTING:
- If the website is slow, try during off-peak hours
- You may need to download crops one at a time if bulk download fails
- Save the file with EXACT name: fao_crop_yield_raw.csv
"""
    
    print(instructions)
    
    # Create a placeholder file
    placeholder_path = Config.AGRICULTURE_DIR / "README_FAO_DOWNLOAD.txt"
    with open(placeholder_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"Instructions saved to: {placeholder_path}")
    print("REMINDER: You must manually download FAO crop yield data!")
    print("   Follow the instructions above or check the README file")

# ============================================================================
# 4. CREATE SUMMARY REPORT
# ============================================================================

def create_summary_report(co2_df, temp_df, rain_df, humid_df):
    """Create a summary report of downloaded data"""
    print("\n" + "="*70)
    print("DATA DOWNLOAD SUMMARY")
    print("="*70)
    
    # Create summary dictionary
    summary_parts = []
    summary_parts.append("\nDOWNLOAD SUMMARY REPORT")
    summary_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_parts.append(f"Period: {Config.START_YEAR}-{Config.END_YEAR}")
    total_states = sum(len(states) for states in Config.ZONES.values())
    summary_parts.append(f"Coverage: {len(Config.ZONES)} zones, {total_states} states\n")
    
    summary_parts.append("="*70)
    summary_parts.append("DOWNLOADED DATA FILES:")
    summary_parts.append("="*70)
    
    # Use ASCII-friendly indicators
    check_mark = "[OK]"
    cross_mark = "[FAILED]"
    warning_mark = "[WARNING]"
    
    # CO2 Data
    if co2_df is not None:
        sources = co2_df['Source'].value_counts().to_dict() if 'Source' in co2_df.columns else {}
        summary_parts.append(f"\n{check_mark} CO2 DATA (NASA OCO-2)")
        summary_parts.append(f"   File: climate/co2_data.csv")
        summary_parts.append(f"   Records: {len(co2_df):,} (per state-month)")
        summary_parts.append(f"   Period: {co2_df['Year'].min()}-{co2_df['Year'].max()}")
        summary_parts.append(f"   Range: {co2_df['CO2_ppm'].min():.2f} - {co2_df['CO2_ppm'].max():.2f} ppm")
        summary_parts.append(f"   Sources: {sources}")
    else:
        summary_parts.append(f"\n{cross_mark} CO2 DATA - DOWNLOAD FAILED")
    
    # Temperature Data
    if temp_df is not None:
        summary_parts.append(f"\n{check_mark} TEMPERATURE DATA (NASA POWER)")
        summary_parts.append(f"   File: climate/temperature_data.csv")
        summary_parts.append(f"   Records: {len(temp_df):,}")
        summary_parts.append(f"   States: {temp_df['State'].nunique()}")
        summary_parts.append(f"   Avg Temp Range: {temp_df['Avg_Temp_C'].min():.1f}C - {temp_df['Avg_Temp_C'].max():.1f}C")
    else:
        summary_parts.append(f"\n{cross_mark} TEMPERATURE DATA - DOWNLOAD FAILED")
    
    # Rainfall Data
    if rain_df is not None:
        summary_parts.append(f"\n{check_mark} RAINFALL DATA (NASA POWER)")
        summary_parts.append(f"   File: climate/rainfall_data.csv")
        summary_parts.append(f"   Records: {len(rain_df):,}")
        summary_parts.append(f"   States: {rain_df['State'].nunique()}")
        summary_parts.append(f"   Rainfall Range: {rain_df['Rainfall_mm'].min():.1f}mm - {rain_df['Rainfall_mm'].max():.1f}mm")
    else:
        summary_parts.append(f"\n{cross_mark} RAINFALL DATA - DOWNLOAD FAILED")
    
    # Humidity Data
    if humid_df is not None:
        summary_parts.append(f"\n{check_mark} HUMIDITY DATA (NASA POWER)")
        summary_parts.append(f"   File: climate/humidity_data.csv")
        summary_parts.append(f"   Records: {len(humid_df):,}")
        summary_parts.append(f"   States: {humid_df['State'].nunique()}")
        summary_parts.append(f"   Humidity Range: {humid_df['Avg_Humidity_Percent'].min():.1f}% - {humid_df['Avg_Humidity_Percent'].max():.1f}%")
    else:
        summary_parts.append(f"\n{cross_mark} HUMIDITY DATA - DOWNLOAD FAILED")
    
    # Manual download needed
    summary_parts.append(f"\n{warning_mark} CROP YIELD DATA (FAO) - MANUAL DOWNLOAD REQUIRED")
    summary_parts.append(f"   Follow instructions in: agriculture/README_FAO_DOWNLOAD.txt")
    summary_parts.append(f"   Target file: agriculture/fao_crop_yield_raw.csv")
    
    summary_parts.append("\n" + "="*70)
    summary_parts.append("NEXT STEPS:")
    summary_parts.append("="*70)
    summary_parts.append("1. Download FAO crop yield data (see instructions above)")
    summary_parts.append("2. Run data preprocessing script to create master datasets")
    summary_parts.append("3. Begin model training with FNN, LSTM, and Hybrid models")
    
    summary_text = "\n".join(summary_parts)
    
    # Print pretty version to console
    print_pretty_summary(co2_df, temp_df, rain_df, humid_df)
    
    # Save ASCII version to file
    summary_file = Config.BASE_DIR / "DOWNLOAD_SUMMARY.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return summary_file

def print_pretty_summary(co2_df, temp_df, rain_df, humid_df):
    """Print pretty summary to console with symbols"""
    print("\nDOWNLOAD SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Period: {Config.START_YEAR}-{Config.END_YEAR}")
    total_states = sum(len(states) for states in Config.ZONES.values())
    print(f"Coverage: {len(Config.ZONES)} zones, {total_states} states\n")
    
    print("="*70)
    print("DOWNLOADED DATA FILES:")
    print("="*70)
    
    # CO2 Data
    if co2_df is not None:
        sources = co2_df['Source'].value_counts().to_dict() if 'Source' in co2_df.columns else {}
        print(f"\n✓ CO2 DATA (NASA OCO-2)")
        print(f"   File: climate/co2_data.csv")
        print(f"   Records: {len(co2_df):,} (per state-month)")
        print(f"   Period: {co2_df['Year'].min()}-{co2_df['Year'].max()}")
        print(f"   Range: {co2_df['CO2_ppm'].min():.2f} - {co2_df['CO2_ppm'].max():.2f} ppm")
        print(f"   Sources: {sources}")
    else:
        print(f"\n✗ CO2 DATA - DOWNLOAD FAILED")
    
    # Temperature Data
    if temp_df is not None:
        print(f"\n✓ TEMPERATURE DATA (NASA POWER)")
        print(f"   File: climate/temperature_data.csv")
        print(f"   Records: {len(temp_df):,}")
        print(f"   States: {temp_df['State'].nunique()}")
        print(f"   Avg Temp Range: {temp_df['Avg_Temp_C'].min():.1f}°C - {temp_df['Avg_Temp_C'].max():.1f}°C")
    else:
        print(f"\n✗ TEMPERATURE DATA - DOWNLOAD FAILED")
    
    # Rainfall Data
    if rain_df is not None:
        print(f"\n✓ RAINFALL DATA (NASA POWER)")
        print(f"   File: climate/rainfall_data.csv")
        print(f"   Records: {len(rain_df):,}")
        print(f"   States: {rain_df['State'].nunique()}")
        print(f"   Rainfall Range: {rain_df['Rainfall_mm'].min():.1f}mm - {rain_df['Rainfall_mm'].max():.1f}mm")
    else:
        print(f"\n✗ RAINFALL DATA - DOWNLOAD FAILED")
    
    # Humidity Data
    if humid_df is not None:
        print(f"\n✓ HUMIDITY DATA (NASA POWER)")
        print(f"   File: climate/humidity_data.csv")
        print(f"   Records: {len(humid_df):,}")
        print(f"   States: {humid_df['State'].nunique()}")
        print(f"   Humidity Range: {humid_df['Avg_Humidity_Percent'].min():.1f}% - {humid_df['Avg_Humidity_Percent'].max():.1f}%")
    else:
        print(f"\n✗ HUMIDITY DATA - DOWNLOAD FAILED")
    
    # Manual download needed
    print(f"\n⚠ CROP YIELD DATA (FAO) - MANUAL DOWNLOAD REQUIRED")
    print(f"   Follow instructions in: agriculture/README_FAO_DOWNLOAD.txt")
    print(f"   Target file: agriculture/fao_crop_yield_raw.csv")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Download FAO crop yield data (see instructions above)")
    print("2. Run data preprocessing script to create master datasets")
    print("3. Begin model training with FNN, LSTM, and Hybrid models")

# ============================================================================
# 5. ALTERNATIVE: SIMPLE TEST FUNCTION
# ============================================================================

def test_nasa_api():
    """Test NASA POWER API with a simple request"""
    print("\n" + "="*70)
    print("TESTING NASA POWER API")
    print("="*70)
    
    # Test with Kaduna coordinates
    test_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'parameters': 'T2M',
        'community': 'AG',
        'longitude': 7.44,
        'latitude': 10.52,
        'start': '20200101',
        'end': '20200110',
        'format': 'JSON'
    }
    
    try:
        print("Testing API with simple request...")
        response = requests.get(test_url, params=params, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("API is working!")
            print(f"Data keys: {list(data.keys())}")
            
            if 'properties' in data and 'parameter' in data['properties']:
                params = data['properties']['parameter']
                print(f"Parameters returned: {list(params.keys())}")
                
                if 'T2M' in params:
                    temp_data = params['T2M']
                    print(f"Sample temperature data (first 5 days):")
                    for i, (date, temp) in enumerate(list(temp_data.items())[:5]):
                        print(f"  {date}: {temp}°C")
                    
                    return True
        else:
            print(f"API returned error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"Error testing API: {str(e)}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    try:
        # Print header
        print_header()
        
        # Test NASA API first (optional)
        print("Testing NASA POWER API connectivity...")
        api_working = test_nasa_api()
        
        if not api_working:
            print("\nWARNING: NASA POWER API test failed!")
            print("   The script may not be able to download climate data.")
            print("   Please check:")
            print("   1. Your internet connection")
            print("   2. NASA POWER API status (https://power.larc.nasa.gov/)")
            print("   3. API parameters are correct")
            
            proceed = input("\nDo you want to continue anyway? (yes/no): ")
            if proceed.lower() != 'yes':
                print("Script terminated.")
                return
        
        # Create directories
        print("\nPreparing directories...")
        create_directories()
        print()
        
        # Download CO2 data
        co2_df = download_co2_data()
        
        # Download NASA POWER climate data
        temp_df, rain_df, humid_df = collect_all_nasa_climate_data()
        
        # Create FAO download instructions
        create_fao_download_instructions()
        
        # Create summary report
        summary_file = create_summary_report(co2_df, temp_df, rain_df, humid_df)
        
        # Final message
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"\nAll files saved in: {Config.BASE_DIR}")
        print(f"Summary report: {summary_file}")
        
        print("\n" + "="*70)
        print("NEXT ACTIONS REQUIRED:")
        print("="*70)
        print("1. MANUALLY download FAO crop yield data")
        print("   See: project_data/raw_data/agriculture/README_FAO_DOWNLOAD.txt")
        print("\n2. After FAO download, run preprocessing script")
        print("   This will create the final datasets for analysis")
        print("\n3. Begin model training and analysis")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        print("   You can re-run this script to continue")
        print("   Partial downloads have been saved")
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {str(e)}")
        print("   Error details:")
        traceback.print_exc()
        print("\n   Please check your internet connection and try again")
        print("   If the problem persists, contact support")

if __name__ == "__main__":
    # Check internet connectivity
    try:
        print("Checking internet connection...", end=" ")
        requests.get("https://www.google.com", timeout=5)
        print("Connected")
        main()
    except requests.exceptions.ConnectionError:
        print("No internet connection")
        print("Please connect to the internet and run the script again")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your network settings and try again")