# Kelvin to Celsius conversion
KELVIN_TO_CELSIUS = -273.15

# WMO standard climate reference period 1991-2020
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"

# 2023 coral bleaching event — worst on record globally
ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2023-12-31"

# Climatology threshold (Hobday et al. 2016): percentile of all baseline SST within
# ±half-width days of each day of year, then a circular running mean over the year
CLIMATOLOGY_PERCENTILE = 0.90
CLIMATOLOGY_WINDOW_HALF_WIDTH_DAYS = 5
CLIMATOLOGY_SMOOTHING_WINDOW_DAYS = 31

# MHW detection: minimum consecutive days above threshold
MHW_PERSISTENCE_DAYS = 5

# MHW detection: events separated by gaps of at most this many days are joined
# into one event, gap days included (Hobday et al. 2016)
MHW_MAX_GAP_DAYS = 2

# DHW: minimum HotSpot (SST - MMM, °C) that is accumulated into DHW
BLEACHING_THRESHOLD_OFFSET = 1.0

# DHW: rolling window in days (12 weeks), NOAA Coral Reef Watch methodology
DHW_WINDOW_DAYS = 84

# zarr storage format version
ZARR_FORMAT = 2