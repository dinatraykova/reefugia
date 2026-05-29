import os
import zarr
import xarray as xr
from reefwatch import fetch

BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"

def get_climatology_path(region_name: str, start: str, end: str) -> str:
    start_year = start[:4]
    end_year = end[:4]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", f"climatology_{region_name}_{start_year}_{end_year}.zarr")

def load_or_compute_climatology(region_name:str) -> xr.Dataset:
    path = get_climatology_path(region_name, BASELINE_START, BASELINE_END)
    if os.path.exists(path):
        return xr.open_zarr(path, zarr_format=2, consolidated=True)
    else:
        historical = fetch.fetch_data(region_name, BASELINE_START, BASELINE_END)
        threshold = historical["analysed_sst"].groupby("time.dayofyear").quantile(0.90)
        climatology = threshold.to_dataset(name="sst_threshold")
        climatology.to_zarr(path, zarr_format=2)
        zarr.consolidate_metadata(path)
        return climatology