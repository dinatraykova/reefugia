"""Validation of reefugia's outputs against reference products."""
import numpy as np
import xarray as xr


def compare_dhw(dhw: xr.DataArray, reference: xr.DataArray, ocean_mask: xr.DataArray) -> dict:
    """Compare DHW with a reference product (e.g. NOAA CRW) over shared dates and ocean pixels."""
    # daily products stamp days differently (00:00 vs 12:00)
    dhw = dhw.assign_coords(time=dhw.time.dt.floor("D"))
    ref = reference.assign_coords(time=reference.time.dt.floor("D"))

    # match reference pixels to ours (nearest centre, within about half a 0.05° pixel)
    ref = ref.sortby("latitude").reindex(
        latitude=dhw.latitude, longitude=dhw.longitude, method="nearest", tolerance=0.03)
    dhw, ref = xr.align(dhw, ref, join="inner")

    valid = (ocean_mask & ref.notnull().any(dim="time")).compute()
    peak = dhw.max(dim="time").where(valid).compute()
    ref_peak = ref.max(dim="time").where(valid).compute()
    diff = peak - ref_peak

    a, b = peak.values.ravel(), ref_peak.values.ravel()
    keep = ~np.isnan(a) & ~np.isnan(b)
    ts = dhw.where(valid).mean(dim=["latitude", "longitude"]).compute()
    ref_ts = ref.where(valid).mean(dim=["latitude", "longitude"]).compute()

    return {
        "n_pixels": int(keep.sum()),
        "n_days": int(dhw.sizes["time"]),
        "r_spatial": float(np.corrcoef(a[keep], b[keep])[0, 1]),
        "r_timeseries": float(np.corrcoef(ts.values, ref_ts.values)[0, 1]),
        "mean_peak": float(a[keep].mean()),
        "mean_peak_reference": float(b[keep].mean()),
        "mean_bias": float((a[keep] - b[keep]).mean()),
        "rmse": float(np.sqrt(((a[keep] - b[keep]) ** 2).mean())),
        "diff": diff,
        "timeseries": ts,
        "timeseries_reference": ref_ts,
    }
