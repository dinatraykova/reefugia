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

def get_climatology_path(region_name: str, start: str, end: str) -> str:
    start_year = start[:4]
    end_year = end[:4]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"climatology_{region_name}_{start_year}_{end_year}.zarr")

def load_or_compute_climatology(region_name:str) -> xr.Dataset:
    """Load pre-computed climatology from zarr cache or compute from scratch.
    
    Uses WMO standard 1991-2020 baseline period.
    Threshold is the 90th percentile SST for each day of year and grid point,
    smoothed with an 11-day rolling window following Hobday et al. (2016).
    """
    path = get_climatology_path(region_name, BASELINE_START, BASELINE_END)
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
    else:
        historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
        threshold = historical["analysed_sst"].groupby("time.dayofyear").quantile(CLIMATOLOGY_PERCENTILE)
        # smooth threshold with rolling window as per Hobday et al. (2016)
        threshold = threshold.rolling(dayofyear=CLIMATOLOGY_SMOOTHING_WINDOW_DAYS, center=True, min_periods=1).mean()

        climatology = threshold.to_dataset(name="sst_threshold")

        # rechunk to uniform sizes required by zarr
        climatology = climatology.chunk("auto")
        climatology.to_zarr(path, zarr_format=ZARR_FORMAT)
        zarr.consolidate_metadata(path)
        return climatology
    
def detect_mhw(sst: xr.DataArray, region_name: str) -> dict:
    """Detect marine heatwave exceedance for each grid point and time step.

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

def compute_mmm(region_name: str) -> xr.DataArray:
    """Compute Max Monthly Mean SST — the NOAA coral bleaching threshold baseline"""
    historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
    monthly_mean = historical["analysed_sst"].groupby("time.month").mean()
    mmm = monthly_mean.max(dim="month")
    return mmm

def compute_dhw(sst: xr.DataArray, region_name: str) -> xr.DataArray:
    """Compute Degree Heating Weeks following NOAA Coral Reef Watch methodology.
    
    DHW measures accumulated heat stress above the bleaching threshold over
    a 12-week rolling window. Bleaching threshold = MMM + 1°.

    DHW >= 4  : bleaching likely
    DHW >= 8  : severe bleaching and mortality likely

    sst — the "analysed_sst" DataArray from fetch_data(), e.g. ds["analysed_sst"].
    """
    # compute the Maximum Monthly Mean for this region
    mmm = compute_mmm(region_name)

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

# TODO: add MHW event counting (n_events, mean_duration) per pixel
# Reference: marineHeatWaves package by Hobday et al.
# Challenge: apply_ufunc with vectorize=True is slow for large grids