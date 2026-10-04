# reefugia 🪸
A simple open-source Python toolkit for detecting marine heatwave (MHW)
conditions and coral bleaching thermal stress in the Coral Triangle, 
Andaman Sea, Red Sea or the Great Barrier Reef using Copernicus Marine 
Service data.

## Motivation 🐠
Despite covering only a small fraction of the sea floor, coral reefs are 
home to over a quarter of marine life. 
Marine heatwaves are periods of anomalously warm sea surface temperature 
and are the primary cause of mass coral bleaching events. 
*reefugia* (from *refugia* — ecological safe havens where species survive 
environmental stress) is a simple, open-source pipeline for detecting and 
analysing thermal stress events using public satellite and reanalysis data.

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
reefugia implements the [NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/main/)
methodology for computing DHWs — a measure of accumulated thermal 
stress above the local maximum monthly mean temperature. 

- DHW ≥ 4: coral bleaching likely
- DHW ≥ 8: severe bleaching and mortality expected
- DHW ≥ 12: multi-species mortality likely 
- DHW ≥ 16: severe multi-species mortality (>50% of corals) 
- DHW ≥ 20: near complete mortality (>80% of corals) likely

### References

- Hobday et al. (2016) — A hierarchical approach to defining marine heatwaves. 
  *Progress in Oceanography*, 141, 227-238. 
  DOI: [10.1016/j.pocean.2015.12.014](https://doi.org/10.1016/j.pocean.2015.12.014)

## Pipeline

reefugia computes the following from raw SST:

1. **Climatology** — 90th percentile SST threshold for each day of year, smoothed
   with an 11-day rolling window (Hobday et al. 2016), computed over the 1991–2020
   WMO baseline period
2. **Marine Heatwave (MHW) detection** — days exceeding the climatological threshold
   for at least 5 consecutive days
3. **Degree Heating Weeks (DHW)** — accumulated heat stress above the bleaching
   threshold (MMM + 1°C) over a 12-week rolling window, following NOAA CRW methodology

All intermediate products (climatology, MHW, DHW) are cached as Zarr stores so
subsequent runs are fast.

## Outputs

For each run, reefugia saves to `outputs/`:

**Data**
- `mhw_days_{region}_{start}_{end}.nc` — spatial map of total MHW days
- `dhw_{region}_{start}_{end}.nc` — daily DHW time series (time, lat, lon)
- `mhw_summary_{region}_{start}_{end}.json` — MHW summary statistics
- `dhw_summary_{region}_{start}_{end}.json` — DHW summary statistics
- `sst_anomaly_{region}_{start}_{end}.csv` — monthly SST anomaly table

**Plots**
- `mhw_days_{region}_{start}_{end}.png` — spatial map of MHW days with hotspot
- `dhw_{region}_{start}_{end}.png` — spatial map of max DHW with hotspot
- `sst_anomaly_max_{region}_{start}_{end}.png` — spatial map of max SST anomaly
- `monthly_matrix_dhw_{region}_{start}_{end}.png` — monthly DHW bar chart (single year) or matrix (multi-year)
- `dhw_animation_{region}_{start}_{end}.mp4` — daily DHW animation

## Default Analysis Period
The default time range (2023-01-01 to 2023-12-31) corresponds to the 
fourth global mass coral bleaching event — the worst on record. The 
Coral Triangle experienced severe thermal stress during this period, 
making it an ideal test case for reefugia's detection pipeline.

## Installation
```bash
git clone https://github.com/dinatraykova/reefugia
cd reefugia
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Jupyter notebooks

To use the notebooks with the correct environment:

```bash
pip install ipykernel
python -m ipykernel install --user --name=reefugia --display-name="reefugia"
jupyter notebook
```

Then select the **reefugia** kernel in the top right of the notebook.

## Credentials
reefugia requires a free Copernicus Marine Service account.
Register at https://marine.copernicus.eu, then run:

```bash
copernicusmarine login
```

This stores your credentials locally and only needs to be done once.

## Usage
```bash
python -m reefugia.cli --region coral_triangle --start 2023-01-01 --end 2023-09-01
```