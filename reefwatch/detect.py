import os
import zarr
import xarray as xr

from reefwatch import fetch
from reefwatch.constants import BASELINE_START, BASELINE_END
from reefwatch.constants import (
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

def get_mmm_path(region_name: str) -> str:
    start_year = BASELINE_START[:4]
    end_year = BASELINE_END[:4]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"mmm_{region_name}_{start_year}_{end_year}.zarr")

def get_mhw_path(region_name: str, start: str, end: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"mhw_{region_name}_{start}_{end}.zarr")

def get_dhw_path(region_name: str, start: str, end: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"dhw_{region_name}_{start}_{end}.zarr")

# compute analysis diagnostics
def compute_climatology(region_name:str) -> xr.Dataset:
    """Compute climatology.
    
    Uses WMO standard 1991-2020 baseline period.
    Threshold is the 90th percentile SST for each day of year and grid point,
    smoothed with an 11-day rolling window following Hobday et al. (2016).
    """
    historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
    threshold = historical["analysed_sst"].groupby("time.dayofyear").quantile(CLIMATOLOGY_PERCENTILE)
    # smooth threshold with rolling window as per Hobday et al. (2016)
    threshold = threshold.rolling(dayofyear=CLIMATOLOGY_SMOOTHING_WINDOW_DAYS, center=True, min_periods=1).mean()

    climatology = threshold.to_dataset(name="sst_threshold")

    # rechunk to uniform sizes required by zarr
    climatology = climatology.chunk("auto")
    return climatology

def compute_mmm(region_name: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute Max Monthly Mean SST and monthly climatology from 1991-2020 baseline.
    
    Returns:
        mmm: Maximum Monthly Mean SST (latitude, longitude)
        monthly_climatology: mean SST for each month (month, latitude, longitude)
    """
    historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
    monthly_mean = historical["analysed_sst"].groupby("time.month").mean()
    mmm = monthly_mean.max(dim="month")
    return mmm, monthly_mean

def compute_mhw(sst: xr.DataArray, region_name: str) -> dict:
    """Compute marine heatwave exceedance for each grid point and time step.

    sst — the "analysed_sst" DataArray from fetch_data(), e.g. ds["analysed_sst"].
    """
    climatology = load_or_compute_climatology(region_name)
    doy = sst.time.dt.dayofyear
    threshold = climatology["sst_threshold"].sel(dayofyear=doy, method="nearest")

    is_over_threshold = sst > threshold

    # rolling sum over previous MHW_PERSISTENCE_DAYS days, for each day of the year
    rolling_sum = is_over_threshold.rolling(time=MHW_PERSISTENCE_DAYS, min_periods=1).sum()

    # True where MHW_PERSISTENCE_DAYS consecutive days of exceedance
    mhw = rolling_sum >= MHW_PERSISTENCE_DAYS
    return {
    "exceedance": is_over_threshold,
    "mhw": mhw
    }

def compute_dhw(sst: xr.DataArray, region_name: str) -> xr.DataArray:
    """Compute Degree Heating Weeks following NOAA Coral Reef Watch methodology.
    
    DHW measures accumulated heat stress above the bleaching threshold over
    a 12-week rolling window. Bleaching threshold = MMM + 1°.

    DHW >= 4  : Alert 1 — bleaching likely
    DHW >= 8  : Alert 2 — severe bleaching
    DHW >= 12 : Alert 3 — mass mortality
    DHW >= 16 : Alert 4 — near complete mortality

    sst — the "analysed_sst" DataArray from fetch_data(), e.g. ds["analysed_sst"].
    """
    # compute the Maximum Monthly Mean for this region
    mmm, _ = load_or_compute_mmm(region_name)

    # bleaching threshold is MMM + BLEACHING_THRESHOLD_OFFSET
    # the offset accounts for the fact that corals can tolerate
    # temperatures slightly above their monthly maximum
    bleaching_threshold = mmm + BLEACHING_THRESHOLD_OFFSET

    # hotspot: how many degrees above the bleaching threshold
    # clip at 0 — negative values (below threshold) don't contribute to stress
    hotspot = (sst - bleaching_threshold).clip(min=0)

    # DHW: rolling sum of hotspots over DHW_WINDOW_DAYS (12 weeks)
    # divided by 7 to convert from degree-days to degree-weeks
    dhw = hotspot.rolling(time=DHW_WINDOW_DAYS, min_periods=1).sum() / 7
    
    return dhw

# caching wrappers    
def load_or_compute_climatology(region_name:str) -> xr.Dataset:
    """Load pre-computed climatology from zarr cache or compute
    using compute_climatology(region_name)
    """
    path = get_climatology_path(region_name)
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
    else:
        climatology = compute_climatology(region_name)
        climatology.to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return climatology

def load_or_compute_mmm(region_name: str) -> tuple[xr.DataArray, xr.DataArray]:
    path = get_mmm_path(region_name)
    if os.path.exists(path):
        ds = xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
        return ds["mmm"], ds["monthly_climatology"]
    else:
        mmm, monthly_mean = compute_mmm(region_name)
        xr.Dataset({
            "mmm": mmm,
            "monthly_climatology": monthly_mean
        }).chunk("auto").to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return mmm, monthly_mean

def load_or_compute_mhw(sst: xr.DataArray, region_name: str, start: str, end: str) -> dict:
    """Load pre-computed MHW results from zarr cache
        or compute using compute_mhw(sst, region_name)
    """
    path = get_mhw_path(region_name, start, end)
    if os.path.exists(path):
        ds = xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
        return {
            "mhw": ds["mhw"],
            "exceedance": ds["exceedance"]
        }
    else:
        result = compute_mhw(sst, region_name)
        mhw_ds = xr.Dataset({
            "mhw": result["mhw"],
            "exceedance": result["exceedance"]
        }).chunk("auto")

        mhw_ds.to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return {
            "mhw": result["mhw"],
            "exceedance": result["exceedance"]
        }

def load_or_compute_dhw(sst: xr.DataArray, region_name: str, start: str, end: str) -> xr.DataArray:
    """Load pre-computed DHW results from zarr cache
            or compute using compute_dhw(sst, region_name)"""
    path = get_dhw_path(region_name, start, end)
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)["dhw"]
    else:
        dhw = compute_dhw(sst, region_name)
        dhw.to_dataset(name="dhw").chunk("auto").to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return dhw

# TODO: add MHW event counting (n_events, mean_duration) per pixel
# Reference: marineHeatWaves package by Hobday et al.
# Challenge: apply_ufunc with vectorize=True is slow for large grids