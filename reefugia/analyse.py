import xarray as xr
import numpy as np
import calendar
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from reefugia.constants import KELVIN_TO_CELSIUS

# robust extremes: the top 1% of ocean pixels, so a few bad coastal pixels can't set them
TOP_QUANTILE = 0.99


def hotspot(field: xr.DataArray, q: float = TOP_QUANTILE) -> tuple[float, float]:
    """(lat, lon) of the hottest area: the top-1% pixel nearest the centre of all
    top-1% pixels, so one outlier pixel can't move it."""
    threshold = float(field.quantile(q))
    top = field.where(field >= threshold).stack(pixel=("latitude", "longitude")).dropna("pixel")
    lat, lon = top.latitude.values, top.longitude.values
    nearest = np.argmin((lat - lat.mean()) ** 2 + (lon - lon.mean()) ** 2)
    return float(lat[nearest]), float(lon[nearest])


def summarise_mhw(mhw_days: xr.DataArray, ocean_mask: xr.DataArray) -> dict:
    """Ocean-only summary stats of MHW days per pixel."""

    hotspot_lat, hotspot_lon = hotspot(mhw_days.where(ocean_mask))
    mhw_counts = mhw_days.where(ocean_mask).values.flatten()
    mhw_counts = mhw_counts[~np.isnan(mhw_counts)]

    return {
    "n_ocean_pixels": len(mhw_counts),
    "mean_days": float(mhw_counts.mean()),
    "median_days": float(np.median(mhw_counts)),
    "p99_days": float(np.quantile(mhw_counts, TOP_QUANTILE)),
    "max_days": float(mhw_counts.max()),
    "pct_affected": float((mhw_counts > 0).mean() * 100),
    "hotspot_lat": hotspot_lat,
    "hotspot_lon": hotspot_lon,
    }

def summarise_dhw(dhw: xr.DataArray, ocean_mask: xr.DataArray) -> dict:
    """Ocean-only summary stats of peak DHW per pixel."""

    dhw_max = dhw.max(dim='time')
    hotspot_lat, hotspot_lon = hotspot(dhw_max.where(ocean_mask))
    dhw_values = dhw_max.where(ocean_mask).values.flatten()
    dhw_values = dhw_values[~np.isnan(dhw_values)]

    return{
        "n_ocean_pixels": len(dhw_values),
        "p99_dhw": float(np.quantile(dhw_values, TOP_QUANTILE)),
        "max_dhw": float(dhw_values.max()),
        "mean_peak_dhw": float(dhw_values.mean()),
        "pct_above_4": float((dhw_values >= 4).mean() * 100),
        "pct_above_8": float((dhw_values >= 8).mean() * 100),
        "hotspot_lat": hotspot_lat,
        "hotspot_lon": hotspot_lon,
        }

def monthly_sst_anomaly(sst: xr.DataArray, monthly_climatology: xr.DataArray, ocean_mask: xr.DataArray) -> pd.DataFrame:
    """Monthly regional mean SST vs the 1991–2020 climatology (°C)."""

    sst_monthly = (sst.where(ocean_mask)
                     .mean(dim=["latitude", "longitude"])
                     .groupby("time.month")
                     .mean()
                     .compute())

    hist_monthly = (monthly_climatology.where(ocean_mask)
                                       .mean(dim=["latitude", "longitude"])
                                       .compute())

    months_present = sst_monthly.month.values
    df = pd.DataFrame({
        "sst_mean": sst_monthly.values + KELVIN_TO_CELSIUS,
        "climatology_mean": hist_monthly.sel(month=months_present).values + KELVIN_TO_CELSIUS,
    }, index=[calendar.month_name[m] for m in months_present])
    df.index.name = "month"
    df["anomaly"] = df["sst_mean"] - df["climatology_mean"]

    return df
