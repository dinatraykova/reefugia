# reefwatch 🪸
An open-source Python toolkit for detecting marine heatwave conditions 
and coral bleaching thermal stress in the Coral Triangle, using 
Copernicus Marine Service data.

## Motivation 🐠
Coral reefs support 25% of all marine life despite covering less than 
1% of the ocean floor. Marine heatwaves — periods of anomalously warm 
sea surface temperature — are the primary driver of mass coral bleaching 
events. reefwatch provides a transparent, reproducible pipeline for 
detecting and analysing these thermal stress events using open satellite 
and reanalysis data.

## Data Sources 🌊
All data is sourced from the [Copernicus Marine Service](https://marine.copernicus.eu/), 
which provides free, open access to ocean data with no quotas.

| Variable | Dataset | Resolution |
|---|---|---|
| Sea Surface Temperature (SST) | Global Ocean SST | 0.05° |
| Ocean Currents | Global Ocean Physics | 0.083° |
| Mixed Layer Depth | Global Ocean Physics | 0.083° |
| Heat Content | Global Ocean Heat Content | 0.25° |

### Key Metric: Degree Heating Weeks (DHW)
reefwatch implements the NOAA Coral Reef Watch methodology for computing 
Degree Heating Weeks — a measure of accumulated thermal stress above the 
local maximum monthly mean temperature. 

- DHW ≥ 4: coral bleaching likely
- DHW ≥ 8: severe bleaching and mortality expected

## Default Analysis Period
The default time range (2023-01-01 to 2023-09-01) corresponds to the 
fourth global mass coral bleaching event — the worst on record. The 
Coral Triangle experienced severe thermal stress during this period, 
making it an ideal test case for reefwatch's detection pipeline.

## Installation
```bash
git clone https://github.com/dinatraykova/reefwatch
cd reefwatch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Credentials
reefwatch requires a free Copernicus Marine Service account.
Register at https://marine.copernicus.eu, then run:

```bash
copernicusmarine login
```

This stores your credentials locally and only needs to be done once.

## Usage
```bash
python -m reefwatch.cli --region coral_triangle --start 2023-01-01 --end 2023-09-01
```