# Kelvin to Celsius conversion
KELVIN_TO_CELSIUS = -273.15

# WMO standard climate reference period 1991-2020
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"

# 2023 coral bleaching event — worst on record globally
ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2023-12-31"

# Climatology threshold: percentile of SST per day-of-year and grid point,
# smoothed with a rolling window of 11 days
CLIMATOLOGY_PERCENTILE = 0.90
CLIMATOLOGY_SMOOTHING_WINDOW_DAYS = 11

# MHW detection: minimum consecutive days above threshold
MHW_PERSISTENCE_DAYS = 5

# DHW: minimum HotSpot (SST - MMM, °C) that is accumulated into DHW
BLEACHING_THRESHOLD_OFFSET = 1.0

# DHW: rolling window in days (12 weeks), NOAA Coral Reef Watch methodology
DHW_WINDOW_DAYS = 84

# zarr storage format version
ZARR_FORMAT = 2