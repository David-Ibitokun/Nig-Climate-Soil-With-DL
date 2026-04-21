#!/usr/bin/env python3
"""Download only CO2 data for Nigeria using the existing climate download module."""

from pathlib import Path
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import download_climate_data as dcd


def enable_network_retries(max_retries=5, backoff_factor=0.5):
    """Enable robust retry for every requests.get call in download_climate_data."""
    retry_strategy = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=backoff_factor,
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Replace module-level requests.get with a retry-enabled session.get
    safe_session = requests.Session()
    safe_session.mount("https://", adapter)
    safe_session.mount("http://", adapter)

    dcd.requests = requests
    dcd.requests.get = safe_session.get

    # Also patch OCO2 session launcher to use this retry strategy for auth downloads
    original_oco2_session = dcd._oco2_session

    def _oco2_session_with_retries():
        session = original_oco2_session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    dcd._oco2_session = _oco2_session_with_retries


def download_co2_with_retry(attempts=3, pause_second=5):
    """Call download_co2_data with error recovery retries."""
    enable_network_retries(max_retries=5, backoff_factor=1)

    for attempt in range(1, attempts + 1):
        print(f"CO2 download attempt {attempt}/{attempts}")
        co2_df = dcd.download_co2_data()
        if co2_df is not None and not co2_df.empty:
            return co2_df

        print(f"Attempt {attempt} failed or returned empty. Waiting {pause_second} seconds before retry.")
        time.sleep(pause_second)

    return None


def main():
    out_path = Path("project_data/raw_data/climate")
    out_path.mkdir(parents=True, exist_ok=True)
    print("Starting CO2 download (OCO-2)...")

    co2_df = download_co2_with_retry(attempts=5, pause_second=10)
    if co2_df is None or co2_df.empty:
        print("ERROR: CO2 download returned no data after retries")
        return 1

    print(f"CO2 download complete: {len(co2_df)} records")
    print(f"Saved: {out_path / 'co2_data.csv'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
