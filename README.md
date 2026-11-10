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

1. **Climatology** — MHW threshold following Hobday et al. (2016). For each day of
   year, the 90th percentile of all SST within ±5 days of it over the 1991–2020 WMO
   baseline period, smoothed with a 31-day running mean that wraps from December
   into January. It matches the marineHeatWaves reference implementation to
   rounding error on test data
2. **Marine Heatwave (MHW) detection** — runs of at least 5 consecutive days above
   the climatological threshold, with every day of the run counted. Events separated
   by gaps of 2 days or less are joined into one event, following Hobday et al. (2016)
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
Both products are checked against independent reference products for the Red Sea
in 2023, a year of severe thermal stress there. The notebooks are in
`notebooks/validation/`.

### Marine heatwaves
`mhw_vs_marineheatwaves.ipynb` compares reefugia with
[marineHeatWaves](https://github.com/ecjoliver/marineHeatWaves), the reference
implementation of Hobday et al. (2016), at three pixels.

| Pixel | MHW days, reefugia | MHW days, marineHeatWaves | Days agreeing |
|---|---|---|---|
| Most MHW days (24.0°N, 35.8°E) | 276 | 270 | 98.4% |
| Central (20.0°N, 38.5°E) | 139 | 139 | 100% |
| North (27.0°N, 34.5°E) | 55 | 55 | 100% |

SST, thresholds and event rules are identical. The only difference in 2023 is a
single day at the hottest pixel where SST equals the threshold to within rounding,
which reefugia counts as above it and marineHeatWaves doesn't. That day completes
one 5-day event, giving the 6 extra MHW days. The notebook needs a local copy of
marineHeatWaves patched for NumPy 2 (see the notebook).

### Degree Heating Weeks
`dhw_vs_noaa_crw.ipynb` compares reefugia's DHW with
[NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/product/5km/) DHW across the
Red Sea. The timing matches closely, with a correlation of 1.00 for the regional
mean through the year. Peak values are lower on average, 9.5 against 12.6 °C-weeks,
with a spatial correlation of 0.65.

Two input differences explain the gap, and they pull in opposite directions.

- **Baseline.** reefugia's maximum monthly mean (MMM) comes from the WMO 1991–2020
  baseline and is 0.56 °C warmer on average than NOAA's older one, which lowers
  DHW everywhere.
- **SST product.** Copernicus C3S SST is 0.22 °C warmer on average than NOAA's
  CoralTemp, mostly in the southern Red Sea, which raises DHW there.

Recomputing reefugia's DHW with NOAA's MMM raises the spatial correlation to 0.79,
and the remaining difference follows the SST difference between the two products.

Given NOAA's own SST and MMM, reefugia's DHW code reproduces NOAA's published DHW
across the whole Red Sea for all of 2023, with correlations of 1.000 in space and
time, zero mean bias and an RMSE of 0.003 °C-weeks (the rounding of NOAA's stored
values). The calculation matches NOAA's exactly, and every difference in the
default output comes from the choice of baseline and SST product.

| Inputs | Spatial r | Mean peak DHW | Mean bias | RMSE |
|---|---|---|---|---|
| reefugia MMM + C3S SST (default) | 0.65 | 9.5 | −3.1 | 4.1 |
| NOAA MMM + C3S SST | 0.79 | 16.9 | +4.3 | 5.4 |
| NOAA MMM + NOAA SST | 1.000 | 12.6 | 0.0 | 0.003 |

NOAA's mean peak DHW is 12.6 °C-weeks. Bias and RMSE are in °C-weeks.

## Tests
Unit tests run the MHW and DHW calculations on small synthetic SST series, with no
network access or cached data needed. They check the MHW threshold and event logic
against plain-loop implementations of the Hobday et al. (2016) rules, and DHW
against hand-worked cases.

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
pip install --group dev    # Jupyter and pytest, needs pip 25.1 or newer
```

With an older pip, install the dev tools by name instead with
`pip install ipykernel jupyterlab pytest`.

### Jupyter notebooks
Jupyter is installed by `uv sync`. To open the notebooks in the browser:

```bash
uv run jupyter lab
```

Without uv, activate the environment first and start Jupyter from it:

```bash
source .venv/bin/activate
jupyter lab
```

To use the environment from a Jupyter installed elsewhere, register it as a
kernel once and then pick **reefugia** in the notebook's kernel menu:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name reefugia --display-name reefugia
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