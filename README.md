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
Sea surface temperature comes from the [Copernicus Marine Service](https://marine.copernicus.eu/), 
which provides free, open access to ocean data with no quotas. NOAA Coral Reef Watch
DHW is used as an independent reference to validate reefugia's DHW.

| Variable | Dataset | Resolution |
|---|---|---|
| Sea Surface Temperature (SST) | Copernicus `C3S-GLO-SST-L4-REP-OBS-SST` | 0.05°, daily |
| Degree Heating Weeks (reference) | [NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/product/5km/) DHW via ERDDAP | 0.05°, daily |

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
- NOAA Coral Reef Watch — [Methodology for the 5 km daily products](https://coralreefwatch.noaa.gov/product/5km/methodology.php)
  (HotSpot and Degree Heating Week definitions)

## Pipeline

reefugia computes the following from raw SST:

1. **Climatology** — 90th percentile SST threshold for each day of year, smoothed
   with an 11-day rolling window (Hobday et al. 2016), computed over the 1991–2020
   WMO baseline period
2. **Marine Heatwave (MHW) detection** — runs of at least 5 consecutive days above
   the climatological threshold, with every day of the run counted. Events separated
   by short gaps are not merged, unlike the full Hobday et al. (2016) definition
3. **Degree Heating Weeks (DHW)** — following NOAA CRW, the daily HotSpot is SST minus
   the maximum monthly mean (MMM) from the 1991–2020 baseline. HotSpots of at least
   1 °C count in full, smaller ones count as zero, and DHW is their sum over a 12-week
   rolling window divided by 7

MHW and DHW use SST from before the start date as a lead-in, so the rolling windows
are full from the first day of the analysis period.

All intermediate products (climatology, MHW, DHW) are cached as Zarr stores in `data/`
so subsequent runs are fast. A cache is recomputed automatically when the settings in
`reefugia/constants.py` or the CLI input dates or region change, but not when the code 
changes, so delete the relevant `data/*.zarr` after changing a calculation.

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
- `sst_anomaly_max_{region}_{start}_{end}.png` — spatial map of the largest monthly
  SST anomaly relative to the 1991–2020 monthly mean
- `monthly_matrix_dhw_{region}_{start}_{end}.png` — monthly DHW bar chart (single year) or matrix (multi-year)
- `dhw_animation_{region}_{start}_{end}.mp4` — daily DHW animation

## Validation
`notebooks/nb03_validate_dhw.ipynb` compares reefugia's DHW for one region and year
with NOAA Coral Reef Watch. It maps the difference in peak DHW per pixel and reports
the spatial and temporal correlation, mean bias and RMSE. Some offset is expected,
since NOAA uses different SST (CoralTemp) and an older, cooler MMM baseline, which
gives larger HotSpots.

## Tests
Unit tests run the MHW and DHW calculations on small synthetic SST series, with no
network access or cached data needed.

```bash
uv run pytest
```

## Default Analysis Period
The default time range (2023-01-01 to 2023-12-31) corresponds to the 
fourth global mass coral bleaching event — the worst on record. The 
Coral Triangle experienced severe thermal stress during this period, 
making it an ideal test case for reefugia's detection pipeline.

## Installation
reefugia uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv, once
git clone https://github.com/dinatraykova/reefugia
cd reefugia
uv sync
```

`uv sync` downloads a suitable Python if needed, creates a `.venv/` virtual
environment, and installs reefugia (in editable mode) with the exact
dependency versions recorded in `uv.lock`, plus the dev tools (`ipykernel`,
`jupyterlab`, `pytest`). Run it again whenever `pyproject.toml` changes.

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Jupyter notebooks
Jupyter is installed by `uv sync`. To open the notebooks in the browser:

```bash
uv run jupyter lab
```

In VS Code, open a notebook and pick the `.venv` interpreter as the kernel.

## Credentials
reefugia requires a free Copernicus Marine Service account.
Register at https://marine.copernicus.eu, then run:

```bash
copernicusmarine login
```

This stores your credentials locally and only needs to be done once.

## Usage
```bash
uv run reefugia --region coral_triangle --start 2023-01-01 --end 2023-09-01
```

`uv run` uses the project environment without activating it. With `.venv`
activated, plain `reefugia ...` (or `python -m reefugia.cli ...`) works the same.