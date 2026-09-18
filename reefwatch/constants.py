# Kelvin to Celsius conversion
KELVIN_TO_CELSIUS = -273.15

# WMO standard climate reference period 1991-2020
# Used by NOAA Coral Reef Watch for DHW calculations
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"

# Time period 2023 coral bleaching event —
# worst on record globally, severely impacted the Coral Triangle
ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2023-12-01"

# Climatology threshold: percentile of SST per day-of-year and grid point,
# smoothed with a rolling window (Hobday et al. 2016)
CLIMATOLOGY_PERCENTILE = 0.90
CLIMATOLOGY_SMOOTHING_WINDOW_DAYS = 11

# MHW detection: minimum consecutive days above threshold (Hobday et al. 2016)
MHW_PERSISTENCE_DAYS = 5

# DHW: bleaching threshold offset above MMM (°C) — accounts for corals'
# tolerance slightly above their monthly maximum
BLEACHING_THRESHOLD_OFFSET = 1.0

# DHW: rolling window in days (12 weeks), NOAA Coral Reef Watch methodology
DHW_WINDOW_DAYS = 84

# zarr storage format version used for climatology cache
ZARR_FORMAT = 2