import os
import warnings
import zarr
import numpy as np
import pandas as pd
import xarray as xr

from reefugia import constants, fetch
from reefugia.constants import BASELINE_START, BASELINE_END
from reefugia.constants import (
    CLIMATOLOGY_PERCENTILE,
    CLIMATOLOGY_WINDOW_HALF_WIDTH_DAYS,
    CLIMATOLOGY_SMOOTHING_WINDOW_DAYS,
    MHW_PERSISTENCE_DAYS,
    MHW_MAX_GAP_DAYS,
    BLEACHING_THRESHOLD_OFFSET,
    DHW_WINDOW_DAYS,
    DHW_MMM_START,
    DHW_MMM_END,
    DHW_MMM_REFERENCE_YEAR,
    ZARR_FORMAT,
)

# cache paths, one set per SST product
def _data_path(name: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", name)

def get_climatology_path(region_name: str) -> str:
    return _data_path(f"climatology_{region_name}_{constants.SST_PRODUCT}.zarr")

def get_mhw_path(region_name: str, start: str, end: str) -> str:
    return _data_path(f"mhw_{region_name}_{constants.SST_PRODUCT}_{start}_{end}.zarr")

def get_dhw_path(region_name: str, start: str, end: str) -> str:
    return _data_path(f"dhw_{region_name}_{constants.SST_PRODUCT}_{start}_{end}.zarr")

# settings stored with each cache; a mismatch on load triggers a recompute
def _mmm_params() -> dict:
    params = {"sst_product": constants.SST_PRODUCT,
              "dhw_mmm_baseline": constants.DHW_MMM_BASELINE}
    if constants.DHW_MMM_BASELINE == "noaa":
        params.update(dhw_mmm_start=DHW_MMM_START, dhw_mmm_end=DHW_MMM_END,
                      dhw_mmm_reference_year=DHW_MMM_REFERENCE_YEAR)
    return params

def _climatology_params() -> dict:
    return {
        **_mmm_params(),
        "baseline_start": BASELINE_START,
        "baseline_end": BASELINE_END,
        "climatology_percentile": CLIMATOLOGY_PERCENTILE,
        "climatology_window_half_width_days": CLIMATOLOGY_WINDOW_HALF_WIDTH_DAYS,
        "climatology_smoothing_window_days": CLIMATOLOGY_SMOOTHING_WINDOW_DAYS,
    }

def _mhw_params() -> dict:
    return {**_climatology_params(),
            "mhw_persistence_days": MHW_PERSISTENCE_DAYS,
            "mhw_max_gap_days": MHW_MAX_GAP_DAYS}

def _dhw_params() -> dict:
    return {
        **_mmm_params(),
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

def _join_gaps(mhw: xr.DataArray, max_gap: int) -> xr.DataArray:
    # a non-MHW day is in a gap of <= max_gap days if there's an MHW day i days
    # before and j days after it with i + j - 1 <= max_gap
    gap = xr.zeros_like(mhw, dtype=bool)
    for i in range(1, max_gap + 1):
        before = mhw.shift(time=i, fill_value=False)
        for j in range(1, max_gap + 2 - i):
            gap = gap | (before & mhw.shift(time=-j, fill_value=False))
    return mhw | gap

def day_of_year(time) -> np.ndarray:
    """Day of year on a leap-year calendar (1–366, Feb 29 = 60, Mar 1 = 61 every year)."""
    t = pd.DatetimeIndex(np.asarray(time))
    late_non_leap = ~t.is_leap_year & (t.month >= 3)
    return t.day_of_year.to_numpy() + late_non_leap.astype(int)


def _hobday_threshold(values: np.ndarray, doy: np.ndarray, percentile: float,
                      half_width: int, smoothing: int) -> np.ndarray:
    # values (..., time) → threshold (..., 366), following marineHeatWaves.detect
    n_time = values.shape[-1]
    offsets = np.arange(-half_width, half_width + 1)
    out = np.full(values.shape[:-1] + (366,), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN land pixels
        for d in range(1, 367):
            if d == 60:
                continue
            centres = np.flatnonzero(doy == d)
            idx = (centres[:, None] + offsets).ravel()
            idx = idx[(idx >= 0) & (idx < n_time)]
            out[..., d - 1] = np.nanpercentile(values[..., idx], percentile * 100, axis=-1)
    # Feb 29 occurs too rarely for its own percentile — interpolate its neighbours
    out[..., 59] = 0.5 * (out[..., 58] + out[..., 60])

    # circular running mean, so the threshold wraps smoothly from Dec into Jan
    h = smoothing // 2
    padded = np.concatenate([out[..., -h:], out, out[..., :h]], axis=-1)
    csum = np.cumsum(np.concatenate([np.zeros_like(padded[..., :1]), padded], axis=-1), axis=-1)
    return (csum[..., smoothing:] - csum[..., :-smoothing]) / smoothing


def compute_threshold(sst: xr.DataArray) -> xr.DataArray:
    """Hobday et al. (2016) MHW threshold per day of year (1–366) from baseline SST."""
    threshold = xr.apply_ufunc(
        _hobday_threshold,
        sst.chunk({"time": -1, "latitude": 32, "longitude": 32}),
        input_core_dims=[["time"]],
        output_core_dims=[["dayofyear"]],
        kwargs={"doy": day_of_year(sst.time.values),
                "percentile": CLIMATOLOGY_PERCENTILE,
                "half_width": CLIMATOLOGY_WINDOW_HALF_WIDTH_DAYS,
                "smoothing": CLIMATOLOGY_SMOOTHING_WINDOW_DAYS},
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"dayofyear": 366}},
    )
    return threshold.assign_coords(dayofyear=np.arange(1, 367)).transpose("dayofyear", ...)


def compute_mmm_noaa(sst: xr.DataArray) -> xr.DataArray:
    """NOAA Coral Reef Watch MMM from daily SST: per pixel and calendar month, a
    linear trend through the yearly monthly means, evaluated at
    DHW_MMM_REFERENCE_YEAR; the MMM is the warmest of the 12 months."""
    monthly = sst.resample(time="MS").mean()

    def at_reference_year(da):
        year = da.time.dt.year.astype(float)
        dx = year - year.mean()
        mean = da.mean(dim="time")
        slope = (dx * (da - mean)).sum(dim="time") / (dx ** 2).sum()
        return mean + slope * (DHW_MMM_REFERENCE_YEAR - year.mean())

    return monthly.groupby("time.month").map(at_reference_year).max(dim="month")


def compute_climatology(region_name: str) -> dict:
    noaa_mmm = constants.DHW_MMM_BASELINE == "noaa"
    start = min(BASELINE_START, DHW_MMM_START) if noaa_mmm else BASELINE_START
    end = max(BASELINE_END, DHW_MMM_END) if noaa_mmm else BASELINE_END
    sst_all = fetch.fetch_data(region_name, start, end)["analysed_sst"]
    sst = sst_all.sel(time=slice(BASELINE_START, BASELINE_END))

    threshold = compute_threshold(sst)

    monthly_mean = sst.groupby("time.month").mean()
    if noaa_mmm:
        mmm = compute_mmm_noaa(sst_all.sel(time=slice(DHW_MMM_START, DHW_MMM_END)))
    else:
        mmm = monthly_mean.max(dim="month")

    return {
        "sst_threshold": threshold,
        "monthly_mean": monthly_mean,
        "mmm": mmm,
    }

def compute_mhw(sst: xr.DataArray, region_name: str) -> dict:
    """MHW exceedance and persistence masks from ds["analysed_sst"]."""
    sst_threshold = load_or_compute_climatology(region_name)["sst_threshold"]

    # lead-in so rolling windows are full and earlier events can be joined
    first = sst.time.values[0]
    sst = _with_lead_in(sst, region_name, MHW_PERSISTENCE_DAYS - 1 + MHW_MAX_GAP_DAYS)

    doy = xr.DataArray(day_of_year(sst.time.values), coords={"time": sst.time}, dims="time")
    threshold = sst_threshold.sel(dayofyear=doy)

    is_over_threshold = sst > threshold

    # single time chunk — rolling over many small chunks is very slow in dask
    is_over_threshold = is_over_threshold.chunk({"time": -1})

    rolling_sum = is_over_threshold.rolling(time=MHW_PERSISTENCE_DAYS, min_periods=1).sum()

    run_end = (rolling_sum >= MHW_PERSISTENCE_DAYS).astype("int8")

    # backward rolling max flags the whole event, not just from day 5
    reverse = slice(None, None, -1)
    mhw = (run_end.isel(time=reverse)
                  .rolling(time=MHW_PERSISTENCE_DAYS, min_periods=1).max()
                  .isel(time=reverse)) > 0

    mhw = _join_gaps(mhw, MHW_MAX_GAP_DAYS)

    # drop the lead-in days
    return {
        "exceedance": is_over_threshold.sel(time=slice(first, None)),
        "mhw": mhw.sel(time=slice(first, None)),
    }

def compute_dhw(sst: xr.DataArray, region_name: str, mmm: xr.DataArray = None,
                lead_in: bool = True) -> xr.DataArray:
    """Degree Heating Weeks (NOAA CRW method) from ds["analysed_sst"].
    Uses the region's cached MMM unless another MMM (in K, same grid) is given.
    With lead_in=False no earlier SST is fetched, so pass SST that starts 83 days
    before the dates you need (the first 83 days have incomplete windows)."""
    if mmm is None:
        mmm = load_or_compute_climatology(region_name)["mmm"]

    # lead-in so rolling windows are full from the first day
    first = sst.time.values[0]
    if lead_in:
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
