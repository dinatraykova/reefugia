import os
import zarr
import pandas as pd
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

# constants each cache depends on — a mismatch on load triggers a recompute
def _climatology_params() -> dict:
    return {
        "baseline_start": BASELINE_START,
        "baseline_end": BASELINE_END,
        "climatology_percentile": CLIMATOLOGY_PERCENTILE,
        "climatology_smoothing_window_days": CLIMATOLOGY_SMOOTHING_WINDOW_DAYS,
    }

def _mhw_params() -> dict:
    return {**_climatology_params(), "mhw_persistence_days": MHW_PERSISTENCE_DAYS}

def _dhw_params() -> dict:
    return {
        "baseline_start": BASELINE_START,
        "baseline_end": BASELINE_END,
        "bleaching_threshold_offset": BLEACHING_THRESHOLD_OFFSET,
        "dhw_window_days": DHW_WINDOW_DAYS,
    }

def _open_valid_cache(path: str, params: dict):
    if not os.path.exists(path):
        return None
    ds = xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)
    cached = {key: ds.attrs.get(key) for key in params}
    if cached != params:
        print(f"Cache {os.path.basename(path)} was computed with different "
              f"settings — recomputing.")
        return None
    return ds

def _write_cache(ds: xr.Dataset, path: str, params: dict) -> None:
    ds.attrs.update(params)
    ds.chunk("auto").to_zarr(path, zarr_format=ZARR_FORMAT, mode="w")
    zarr.consolidate_metadata(path)

def _with_lead_in(sst: xr.DataArray, region_name: str, days: int) -> xr.DataArray:
    first = pd.Timestamp(sst.time.values[0])
    lead = fetch.fetch_data(
        region_name,
        str((first - pd.Timedelta(days=days)).date()),
        str((first - pd.Timedelta(days=1)).date()),
    )["analysed_sst"]
    return xr.concat([lead, sst], dim="time")

# compute analysis diagnostics
def compute_climatology(region_name: str) -> dict:
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
    """MHW exceedance and persistence masks from ds["analysed_sst"]."""
    sst_threshold = load_or_compute_climatology(region_name)["sst_threshold"]

    # lead-in so rolling windows are full from the first day
    first = sst.time.values[0]
    sst = _with_lead_in(sst, region_name, MHW_PERSISTENCE_DAYS - 1)

    doy = sst.time.dt.dayofyear
    threshold = sst_threshold.sel(dayofyear=doy, method="nearest")

    is_over_threshold = sst > threshold

    # single time chunk — rolling over many small chunks is very slow in dask
    is_over_threshold = is_over_threshold.chunk({"time": -1})

    rolling_sum = is_over_threshold.rolling(time=MHW_PERSISTENCE_DAYS, min_periods=1).sum()

    run_end = (rolling_sum >= MHW_PERSISTENCE_DAYS).astype("int8")

    # backward rolling max flags the whole event, not just from day 5 on
    reverse = slice(None, None, -1)
    mhw = (run_end.isel(time=reverse)
                  .rolling(time=MHW_PERSISTENCE_DAYS, min_periods=1).max()
                  .isel(time=reverse)) > 0

    # drop the lead-in days
    return {
        "exceedance": is_over_threshold.sel(time=slice(first, None)),
        "mhw": mhw.sel(time=slice(first, None)),
    }

def compute_dhw(sst: xr.DataArray, region_name: str) -> xr.DataArray:
    """Degree Heating Weeks (NOAA CRW method) from ds["analysed_sst"]."""
    mmm = load_or_compute_climatology(region_name)["mmm"]

    # lead-in so rolling windows are full from the first day
    first = sst.time.values[0]
    sst = _with_lead_in(sst, region_name, DHW_WINDOW_DAYS - 1)

    # NOAA CRW: HotSpots >= 1°C count in full, smaller ones as 0 (NaN kept for land)
    hotspot = sst - mmm
    hotspot = hotspot.where((hotspot >= BLEACHING_THRESHOLD_OFFSET) | hotspot.isnull(), 0)

    # single time chunk — rolling over many small chunks is very slow in dask
    hotspot = hotspot.chunk({"time": -1})

    # degree-days → degree-weeks
    dhw = hotspot.rolling(time=DHW_WINDOW_DAYS, min_periods=1).sum() / 7

    # drop the lead-in days
    return dhw.sel(time=slice(first, None))

# caching wrappers
def load_or_compute_climatology(region_name: str) -> dict:
    """Cached climatology dict: sst_threshold, monthly_mean, mmm."""
    path = get_climatology_path(region_name)
    params = _climatology_params()
    ds = _open_valid_cache(path, params)
    if ds is not None:
        return {
            "sst_threshold": ds["sst_threshold"],
            "monthly_mean": ds["monthly_mean"],
            "mmm": ds["mmm"],
        }
    result = compute_climatology(region_name)
    _write_cache(xr.Dataset({
        "sst_threshold": result["sst_threshold"],
        "monthly_mean": result["monthly_mean"],
        "mmm": result["mmm"],
    }), path, params)
    return result

def load_or_compute_mhw(sst: xr.DataArray, region_name: str, start: str, end: str) -> dict:
    """Cached compute_mhw()."""
    path = get_mhw_path(region_name, start, end)
    params = _mhw_params()
    ds = _open_valid_cache(path, params)
    if ds is not None:
        return {
            "mhw": ds["mhw"],
            "exceedance": ds["exceedance"],
        }
    result = compute_mhw(sst, region_name)
    _write_cache(xr.Dataset({
        "mhw": result["mhw"],
        "exceedance": result["exceedance"],
    }), path, params)
    return result

def load_or_compute_dhw(sst: xr.DataArray, region_name: str, start: str, end: str) -> xr.DataArray:
    """Cached compute_dhw()."""
    path = get_dhw_path(region_name, start, end)
    params = _dhw_params()
    ds = _open_valid_cache(path, params)
    if ds is not None:
        return ds["dhw"]
    dhw = compute_dhw(sst, region_name)
    _write_cache(dhw.to_dataset(name="dhw"), path, params)
    return dhw

# TODO: add MHW event counting (n_events, mean_duration) per pixel
# Challenge: apply_ufunc with vectorize=True is slow for large grids
