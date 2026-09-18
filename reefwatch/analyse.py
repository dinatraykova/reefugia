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

def compare_years(region_name: str, years: list[int], ocean_mask: xr.DataArray) -> pd.DataFrame:
    """Compare MHW statistics across multiple years for a region.
    
    Args:
        region_name: region to analyse
        years:       list of years to compare e.g. [1991, 2023]
        ocean_mask:  boolean DataArray marking ocean pixels as True, land as False
    
    Returns:
        DataFrame with one row per year and columns for key MHW metrics
    """
    rows = []
    for year in years:
        sst = fetch.fetch_data(region_name, f"{year}-01-01", f"{year}-12-31")["analysed_sst"]
        mhw_result = detect.compute_mhw(sst, region_name)
        mhw_days = mhw_result["mhw"].sum(dim="time").compute()
        summary = summarise_mhw(mhw_days, ocean_mask)
        summary["year"] = year
        rows.append(summary)
    
    return pd.DataFrame(rows).set_index("year")

def monthly_sst_anomaly(sst: xr.DataArray, region_name: str, ocean_mask: xr.DataArray) -> pd.DataFrame:
    """Compare monthly mean SST against 30-year climatological baseline.
    
    Args:
        sst:         SST DataArray for analysis period
        region_name: region name for loading historical baseline
        ocean_mask:  boolean DataArray marking ocean pixels as True, land as False
    
    Returns:
        DataFrame with columns: month, sst_mean, climatology_mean (°C), anomaly
    """
    import calendar

    # spatial mean over ocean pixels for each month in analysis period
    sst_monthly = (sst.where(ocean_mask)
                     .mean(dim=["latitude", "longitude"])
                     .groupby("time.month")
                     .mean()
                     .compute())

    # 30 year baseline
    historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
    hist_monthly = (historical["analysed_sst"].where(ocean_mask)
                                              .mean(dim=["latitude", "longitude"])
                                              .groupby("time.month")
                                              .mean()
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