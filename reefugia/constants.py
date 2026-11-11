# Kelvin to Celsius conversion
KELVIN_TO_CELSIUS = -273.15

# SST product for everything: "c3s" (recommended) or "ostia" (DHW runs high,
# see README)
SST_PRODUCT = "c3s"

# WMO standard climate reference period 1991-2020
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"

# 2023: worst global coral bleaching event on record
ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2023-12-31"

# MHW threshold (Hobday et al. 2016)
CLIMATOLOGY_PERCENTILE = 0.90
CLIMATOLOGY_WINDOW_HALF_WIDTH_DAYS = 5
CLIMATOLOGY_SMOOTHING_WINDOW_DAYS = 31

# MHW detection: minimum consecutive days above threshold
MHW_PERSISTENCE_DAYS = 5

# MHW detection: events separated by gaps of at most this many days are joined
MHW_MAX_GAP_DAYS = 2

# DHW: MMM baseline, "noaa" (NOAA CRW method, trend through 1985–2012 monthly
# means evaluated at 1988.29) or "wmo" (warmest 1991–2020 monthly mean)
DHW_MMM_BASELINE = "noaa"
DHW_MMM_START = "1985-01-01"
DHW_MMM_END = "2012-12-31"
DHW_MMM_REFERENCE_YEAR = sum([1985, 1986, 1987, 1988, 1989, 1990, 1993]) / 7  # 1988.2857

# DHW: minimum HotSpot (SST - MMM, °C) that is accumulated into DHW
BLEACHING_THRESHOLD_OFFSET = 1.0

# DHW: rolling window in days (12 weeks), NOAA Coral Reef Watch methodology
DHW_WINDOW_DAYS = 84

# zarr storage format version
ZARR_FORMAT = 2