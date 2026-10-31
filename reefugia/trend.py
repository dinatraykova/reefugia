"""Multi-year DHW trends from NOAA Coral Reef Watch 5 km data."""

import os
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from reefugia.constants import ZARR_FORMAT
from reefugia.fetch import get_region

CRW_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW"


# ── Data access ──────────────────────────────────────────────────────────────

def fetch_crw_dhw(region_name: str, start_year: int = 1986, end_year: int = 2025) -> xr.DataArray:
    """Lazy daily NOAA CRW DHW for a region (record starts 1985-04-01, so 1986 is the first full year)."""
    bounds = get_region(region_name)
    ds = xr.open_dataset(CRW_URL, engine="pydap", chunks={"time": 30})

    # ERDDAP stores NOAA_DHW latitude north → south
    lat = ds.latitude
    if lat[0] > lat[-1]:
        lat_slice = slice(bounds["max_lat"], bounds["min_lat"])
    else:
        lat_slice = slice(bounds["min_lat"], bounds["max_lat"])

    dhw = ds["CRW_DHW"].sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
        latitude=lat_slice,
        longitude=slice(bounds["min_lon"], bounds["max_lon"]),
    )
    return dhw


def _crw_path(region_name: str, start_year: int, end_year: int) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data",
                        f"crw_dhw_annual_max_{region_name}_{start_year}_{end_year}.zarr")


def load_or_fetch_crw_annual_max(region_name: str, start_year: int = 1986,
                                 end_year: int = 2025) -> xr.DataArray:
    """Cached annual max NOAA CRW DHW per pixel (dims: year, latitude, longitude)."""
    import zarr
    path = _crw_path(region_name, start_year, end_year)
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=ZARR_FORMAT, consolidated=True)["dhw"]
    # reduce lazily so only one year per pixel is held in memory, not the daily record
    dhw = annual_max_dhw(fetch_crw_dhw(region_name, start_year, end_year)).compute()
    dhw.to_dataset(name="dhw").to_zarr(path, zarr_format=ZARR_FORMAT)
    zarr.consolidate_metadata(path)
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