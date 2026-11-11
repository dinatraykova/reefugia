"""Multi-year DHW trends from NOAA Coral Reef Watch 5 km data."""

import os
import tempfile
import time
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from reefugia.constants import ZARR_FORMAT
from reefugia.fetch import get_region

# NOAA CRW 5 km daily product on ERDDAP; the second is a mirror, tried if the first fails
CRW_URLS = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW",
    "https://upwell.pfeg.noaa.gov/erddap/griddap/dhw_5km",
)


# ── Data access ──────────────────────────────────────────────────────────────

# servers that refused to connect this session
_unreachable = set()


def _download_crw(region_name: str, start: str, end: str, variable: str,
                  connect_timeout: int = 30, read_timeout: int = 900,
                  attempts: int = 2) -> xr.DataArray:
    # one ERDDAP NetCDF request; slow servers are retried once, unreachable ones skipped
    import requests
    b = get_region(region_name)
    query = (f"{variable}[({start}T12:00:00Z):1:({end}T12:00:00Z)]"
             f"[({b['max_lat']}):1:({b['min_lat']})]"  # latitude is stored north → south
             f"[({b['min_lon']}):1:({b['max_lon']})]")
    errors = []
    for url in [u for u in CRW_URLS if u not in _unreachable]:
        for attempt in range(1, attempts + 1):
            fd, tmp = tempfile.mkstemp(suffix=".nc")
            os.close(fd)
            try:
                with requests.get(f"{url}.nc?{query}", stream=True,
                                  timeout=(connect_timeout, read_timeout)) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for block in r.iter_content(chunk_size=1 << 20):
                            f.write(block)
                with xr.open_dataset(tmp, engine="scipy") as ds:
                    return ds[variable].load()
            except requests.ConnectTimeout as e:
                _unreachable.add(url)
                errors.append(f"  {url}: {type(e).__name__}: {str(e)[:200]}")
                break
            except requests.RequestException as e:
                errors.append(f"  {url} (attempt {attempt}): {type(e).__name__}: {str(e)[:200]}")
                status = getattr(e.response, "status_code", None)
                if status is not None and status < 500:
                    break  # a bad request won't succeed on retry
                if attempt < attempts:
                    time.sleep(30)
            finally:
                os.remove(tmp)
    raise ConnectionError("Could not download from any NOAA CRW server:\n" + "\n".join(errors))


def _download_range(region_name: str, start: str, end: str, variable: str) -> xr.DataArray:
    # small, spaced-out requests, as NOAA asks
    edges = pd.date_range(start, end, freq="QS").union([pd.Timestamp(start)])
    pieces = []
    for k, piece_start in enumerate(edges):
        piece_end = min(pd.Timestamp(end), edges[k + 1] - pd.Timedelta(days=1)) \
            if k + 1 < len(edges) else pd.Timestamp(end)
        if k:
            time.sleep(5)
        pieces.append(_download_crw(region_name, piece_start.strftime("%Y-%m-%d"),
                                    piece_end.strftime("%Y-%m-%d"), variable))
    return xr.concat(pieces, dim="time")


def _data_path(name: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", name)


def _save(da: xr.DataArray, name: str, path: str) -> None:
    import zarr
    da.to_dataset(name=name).to_zarr(path, zarr_format=ZARR_FORMAT, mode="w")
    zarr.consolidate_metadata(path)


def load_or_fetch_crw_daily(region_name: str, start: str, end: str,
                            variable: str = "CRW_DHW") -> xr.DataArray:
    """Cached daily NOAA CRW data for a region and date range (dates as YYYY-MM-DD),
    e.g. CRW_DHW, CRW_SST or CRW_HOTSPOT. Downloads in 3-month pieces."""
    name = variable.lower().removeprefix("crw_")
    path = _data_path(f"crw_{name}_{region_name}_{start}_{end}.zarr")
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)[variable]
    da = _download_range(region_name, start, end, variable)
    _save(da, variable, path)
    return da


def load_or_fetch_crw_annual_max(region_name: str, start_year: int = 1986,
                                 end_year: int = 2025) -> xr.DataArray:
    """Cached annual max NOAA CRW DHW per pixel (dims: year, latitude, longitude).
    Each year is downloaded and cached on its own, so an interrupted run resumes."""
    path = _data_path(f"crw_dhw_annual_max_{region_name}_{start_year}_{end_year}.zarr")
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)["dhw"]

    years = []
    for year in range(start_year, end_year + 1):
        year_path = _data_path(f"crw_dhw_annual_max_{region_name}_{year}.zarr")
        if os.path.exists(year_path):
            year_max = xr.open_zarr(year_path, zarr_format=ZARR_FORMAT, consolidated=True)["dhw"]
        else:
            print(f"Downloading NOAA CRW DHW for {year}...")
            daily = _download_range(region_name, f"{year}-01-01", f"{year}-12-31", "CRW_DHW")
            year_max = daily.max(dim="time")
            _save(year_max, "dhw", year_path)
        years.append(year_max.expand_dims(year=[year]))

    dhw = xr.concat(years, dim="year").sortby("latitude").compute()
    _save(dhw, "dhw", path)
    return dhw


# ── Aggregation ──────────────────────────────────────────────────────────────

def annual_max_dhw(dhw: xr.DataArray) -> xr.DataArray:
    """Annual max DHW per pixel."""
    return dhw.groupby("time.year").max(dim="time")


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da.latitude))
    return da.weighted(weights).mean(dim=["latitude", "longitude"])


# ── Trend fitting ─────────────────────────────────────────────────────────────

def fit_trend(da_annual: xr.DataArray, degree: int = 1) -> dict:
    """Polynomial fit to regional annual max DHW; returns dict incl. model and r2."""
    regional_mean = area_weighted_mean(da_annual).compute()
    years = regional_mean.year.values.astype(float)
    values = regional_mean.values

    mask = ~np.isnan(values)
    X = years[mask].reshape(-1, 1)
    y = values[mask]

    pipeline = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    pipeline.fit(X, y)

    y_fit_full = np.full_like(values, np.nan)
    y_fit_full[mask] = pipeline.predict(X)

    return {
        "degree": degree,
        "years": years,
        "values": values,
        "y_fit": y_fit_full,
        "r2": r2_score(y, pipeline.predict(X)),
        "model": pipeline,
    }


# ── Pixel-level trend map ─────────────────────────────────────────────────────

def pixel_trend(da_annual: xr.DataArray, degree: int = 1) -> xr.DataArray:
    """Per-pixel trend slope in DHW/yr (at the last year if degree > 1)."""
    da_computed = da_annual.compute()
    years = da_computed.year.values.astype(float)
    last_year = years[-1]

    def _pixel_slope(values: np.ndarray) -> float:
        mask = ~np.isnan(values)
        if mask.sum() < max(3, degree + 1):
            return np.nan
        X = years[mask].reshape(-1, 1)
        y = values[mask]
        pipe = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        pipe.fit(X, y)
        if degree == 1:
            return float(pipe.named_steps["linearregression"].coef_[1])
        # numerical derivative at last_year
        h = 0.5
        return float(
            (pipe.predict([[last_year + h]]) - pipe.predict([[last_year - h]]))[0] / (2 * h)
        )

    return xr.apply_ufunc(
        _pixel_slope,
        da_computed,
        input_core_dims=[["year"]],
        vectorize=True,
        output_dtypes=[float],
    )