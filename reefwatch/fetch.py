import xarray as xr
import copernicusmarine

# Bounding boxes for key coral reef regions
# Source: Coral Triangle Initiative (CTI-CFF, 2009), Wikipedia, GBRMPA
# TODO: replace with proper polygon boundaries from regionmask for finer analysis
REGIONS = {
    "coral_triangle": {
        # Official CTI boundary: 23°N to 16°S, 90°E to 175°E
        "min_lon": 110.0,
        "max_lon": 155.0,
        "min_lat": -15.0,
        "max_lat": 20.0,
    },
    "andaman_sea": {
        # Extends 92°E to 100°E, 4°N to 20°N
        "min_lon": 92.0,
        "max_lon": 100.0,
        "min_lat": 5.0,
        "max_lat": 16.0,
    },
        "great_barrier_reef": {
        # Torres Strait (9°S) to Lady Elliot Island (24°S), along Queensland coast
        "min_lon": 142.0,
        "max_lon": 154.0,
        "min_lat": -24.5,
        "max_lat":  -9.0,
    },
    "red_sea": {
        # Bab-el-Mandeb strait (12°N) to Gulf of Suez/Aqaba (30°N)
        "min_lon":  32.0,
        "max_lon":  43.5,
        "min_lat":  12.0,
        "max_lat":  30.0,
    },
}

def get_region(region_name: str) -> dict:
    """Return bounding box coordinates for a named region."""
    if region_name not in REGIONS:
        raise ValueError(f"Unknown region '{region_name}'. Available regions: {list(REGIONS.keys())}")
    return REGIONS[region_name]

def fetch_data(region_name: str, start: str, end: str):
    """Fetch the sea surface temperature data from Copernicus Marine Service."""

    region = get_region(region_name)

    dataset = copernicusmarine.open_dataset(
        dataset_id="C3S-GLO-SST-L4-REP-OBS-SST",
        minimum_longitude=region["min_lon"],
        maximum_longitude=region["max_lon"],
        minimum_latitude=region["min_lat"],
        maximum_latitude=region["max_lat"],
        start_datetime=start,
        end_datetime=end,
        variables=["analysed_sst"],
    )
    return dataset