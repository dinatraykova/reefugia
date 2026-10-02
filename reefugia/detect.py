import os
import zarr
import xarray as xr

from reefugia import fetch
from reefugia.constants import BASELINE_START, BASELINE_END
from reefugia.constants import (
    CLIMATOLOGY_PERCENTILE,
    CLIMATOLOGY_SMOOTHING_WINDOW_DAYS,
    MHW_PERSISTENCE_DAYS,
    BLEACHING_THRESHOLD_OFFSET,
    DHW_WINDOW_DAYS,
    ZARR_FORMAT,
)

# path helpers
def get_climatology_path(region_name: str) -> str:
    start_year = BASELINE_START[:4]
    end_year = BASELINE_END[:4]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"climatology_{region_name}_{start_year}_{end_year}.zarr")

def get_mhw_path(region_name: str, start: str, end: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"mhw_{region_name}_{start}_{end}.zarr")

def get_dhw_path(region_name: str, start: str, end: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"dhw_{region_name}_{start}_{end}.zarr")

# compute analysis diagnostics
def compute_climatology(region_name: str) -> dict:
    """Compute climatology, Max Monthly Mean SST and monthly mean from
    1991-2020 baseline. Single fetch for all three diagnostics.

    Uses WMO standard 1991-2020 baseline period.
    Threshold is the 90th percentile SST for each day of year and grid point,
    smoothed with an 11-day rolling window following Hobday et al. (2016).

    Returns dict with keys:
        sst_threshold: 90th percentile SST threshold (dayofyear, latitude, longitude)
        monthly_mean:  mean SST for each month (month, latitude, longitude)
        mmm:           Maximum Monthly Mean SST (latitude, longitude)
    """
    historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
    sst = historical["analysed_sst"]

    # 90th percentile threshold, smoothed with 11-day rolling window
    threshold = sst.groupby("time.dayofyear").quantile(CLIMATOLOGY_PERCENTILE)
    threshold = threshold.rolling(
        dayofyear=CLIMATOLOGY_SMOOTHING_WINDOW_DAYS, center=True, min_periods=1
    ).mean()

    monthly_mean = sst.groupby("time.month").mean()
    mmm = monthly_mean.max(dim="month")

    return {
        "sst_threshold": threshold,
        "monthly_mean": monthly_mean,
        "mmm": mmm,
    }

def compute_mhw(sst: xr.DataArray, region_name: str) -> dict:
    """Compute marine heatwave exceedance for each grid point and time step.

    sst — the "analysed_sst" DataArray from fetch_data(), e.g. ds["analysed_sst"].
    """
    sst_threshold = load_or_compute_climatology(region_name)["sst_threshold"]
    doy = sst.time.dt.dayofyear
    threshold = sst_threshold.sel(dayofyear=doy, method="nearest")

    is_over_threshold = sst > threshold

    # rolling ops need the full time axis in one chunk, or dask has to stitch
    # together many small overlapping chunks (slow/stalls for large windows)
    is_over_threshold = is_over_threshold.chunk({"time": -1})

    # rolling sum over previous MHW_PERSISTENCE_DAYS days
    rolling_sum = is_over_threshold.rolling(time=MHW_PERSISTENCE_DAYS, min_periods=1).sum()

    # True where MHW_PERSISTENCE_DAYS consecutive days of exceedance
    mhw = rolling_sum >= MHW_PERSISTENCE_DAYS
    return {
        "exceedance": is_over_threshold,
        "mhw": mhw,
    }

def compute_dhw(sst: xr.DataArray, region_name: str) -> xr.DataArray:
    """Compute Degree Heating Weeks following NOAA Coral Reef Watch methodology.

    DHW measures accumulated heat stress above the bleaching threshold over
    a 12-week rolling window. Bleaching threshold = MMM + 1°C.

    DHW >= 4  : Alert 1 — bleaching likely
    DHW >= 8  : Alert 2 — severe bleaching
    DHW >= 12 : Alert 3 — mass mortality
    DHW >= 16 : Alert 4 — near complete mortality

    sst — the "analysed_sst" DataArray from fetch_data(), e.g. ds["analysed_sst"].
    """
    mmm = load_or_compute_climatology(region_name)["mmm"]

    # bleaching threshold: corals can tolerate slightly above their monthly maximum
    bleaching_threshold = mmm + BLEACHING_THRESHOLD_OFFSET

    # hotspot: degrees above bleaching threshold; clip at 0
    hotspot = (sst - bleaching_threshold).clip(min=0)

    # rolling ops need the full time axis in one chunk, or dask has to stitch
    # together many small overlapping chunks (slow/stalls for large windows) —
    # DHW_WINDOW_DAYS=84 makes this far more expensive than MHW's 5-day window
    hotspot = hotspot.chunk({"time": -1})

    # rolling sum over DHW_WINDOW_DAYS (12 weeks), convert degree-days to degree-weeks
    dhw = hotspot.rolling(time=DHW_WINDOW_DAYS, min_periods=1).sum() / 7

    return dhw

# caching wrappers
def load_or_compute_climatology(region_name: str) -> dict:
    """Load pre-computed climatology from zarr cache or compute fresh.

    Returns dict with keys: sst_threshold, monthly_mean, mmm.
    """
    path = get_climatology_path(region_name)
    if os.path.exists(path):
        ds = xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
        return {
            "sst_threshold": ds["sst_threshold"],
            "monthly_mean": ds["monthly_mean"],
            "mmm": ds["mmm"],
        }
    else:
        result = compute_climatology(region_name)
        xr.Dataset({
            "sst_threshold": result["sst_threshold"],
            "monthly_mean": result["monthly_mean"],
            "mmm": result["mmm"],
        }).chunk("auto").to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return result

def load_or_compute_mhw(sst: xr.DataArray, region_name: str, start: str, end: str) -> dict:
    """Load pre-computed MHW results from zarr cache or compute fresh."""
    path = get_mhw_path(region_name, start, end)
    if os.path.exists(path):
        ds = xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
        return {
            "mhw": ds["mhw"],
            "exceedance": ds["exceedance"],
        }
    else:
        result = compute_mhw(sst, region_name)
        xr.Dataset({
            "mhw": result["mhw"],
            "exceedance": result["exceedance"],
        }).chunk("auto").to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return result

def load_or_compute_dhw(sst: xr.DataArray, region_name: str, start: str, end: str) -> xr.DataArray:
    """Load pre-computed DHW results from zarr cache or compute fresh."""
    path = get_dhw_path(region_name, start, end)
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)["dhw"]
    else:
        dhw = compute_dhw(sst, region_name)
        dhw.to_dataset(name="dhw").chunk("auto").to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return dhw

# TODO: add MHW event counting (n_events, mean_duration) per pixel
# Challenge: apply_ufunc with vectorize=True is slow for large grids
