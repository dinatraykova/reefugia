import xarray as xr
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from reefwatch import fetch, detect
from reefwatch.constants import BASELINE_START, BASELINE_END, KELVIN_TO_CELSIUS

def summarise_mhw(mhw_days: xr.DataArray, ocean_mask: xr.DataArray) -> dict:
    """Summarise MHW statistics across the region.
    
    Args:
        mhw_days:   number of MHW days per pixel (latitude, longitude)
        ocean_mask: boolean DataArray marking ocean pixels as True, land as False.
    """

    max_idx = mhw_days.argmax(dim=["latitude", "longitude"])
    mhw_counts = mhw_days.where(ocean_mask).values.flatten()
    mhw_counts = mhw_counts[~np.isnan(mhw_counts)]

    return {
    "n_ocean_pixels": len(mhw_counts),
    "mean_days": float(mhw_counts.mean()),
    "median_days": float(np.median(mhw_counts)),
    "max_days": float(mhw_counts.max()),
    "pct_affected": float((mhw_counts > 0).mean() * 100),
    "hotspot_lat": float(mhw_days.latitude[max_idx["latitude"].values]),
    "hotspot_lon": float(mhw_days.longitude[max_idx["longitude"].values]),
    }

def summarise_dhw(dhw: xr.DataArray, ocean_mask: xr.DataArray) -> dict:
    """Summarise DHW statistics across the region.
    
    Args:
        dhw:        Degree Heating Weeks DataArray (time, latitude, longitude)
        ocean_mask: boolean DataArray marking ocean pixels as True, land as False."""

    dhw_max = dhw.max(dim='time')
    max_idx = dhw_max.argmax(dim=["latitude","longitude"])
    dhw_values = dhw_max.where(ocean_mask).values.flatten()
    dhw_values = dhw_values[~np.isnan(dhw_values)]

    return{
        "n_ocean_pixels": len(dhw_values),
        "max_dhw": float(dhw_values.max()),
        "mean_peak_dhw": float(dhw_values.mean()),
        "pct_above_4": float((dhw_values >= 4).mean() * 100),
        "pct_above_8": float((dhw_values >= 8).mean() * 100),
        "hotspot_lat": float(dhw_max.latitude[max_idx["latitude"].values]),
        "hotspot_lon": float(dhw_max.longitude[max_idx["longitude"].values]),
        }

def monthly_sst_anomaly(sst: xr.DataArray, monthly_climatology: xr.DataArray, ocean_mask: xr.DataArray) -> pd.DataFrame:
    """Compare monthly mean SST against 30-year climatological baseline.
    
    Args:
        sst:                  SST DataArray for analysis period
        monthly_climatology:  monthly mean SST from 1991-2020 baseline (month, lat, lon)
        ocean_mask:           boolean DataArray marking ocean pixels as True
    """
    import calendar

    # spatial mean over ocean pixels for each month in analysis period
    sst_monthly = (sst.where(ocean_mask)
                     .mean(dim=["latitude", "longitude"])
                     .groupby("time.month")
                     .mean()
                     .compute())

    hist_monthly = (monthly_climatology.where(ocean_mask)
                                       .mean(dim=["latitude", "longitude"])
                                       .compute())

    # only use months present in the analysis period
    months_present = sst_monthly.month.values
    df = pd.DataFrame({
        "sst_mean": sst_monthly.values + KELVIN_TO_CELSIUS,
        "climatology_mean": hist_monthly.sel(month=months_present).values + KELVIN_TO_CELSIUS,
    }, index=[calendar.month_name[m] for m in months_present])
    df.index.name = "month"
    df["anomaly"] = df["sst_mean"] - df["climatology_mean"]

    return df