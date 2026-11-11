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
which provides free, open access to ocean data with no quotas. Two SST products are
supported, and the same one is used for every step. NOAA Coral Reef Watch DHW is
used as an independent reference to validate reefugia's DHW.

| Variable | Dataset | Resolution |
|---|---|---|
| SST, default | Copernicus ESA CCI / C3S reprocessed SST, `C3S-GLO-SST-L4-REP-OBS-SST` | 0.05°, daily |
| SST, option | Copernicus OSTIA reprocessed foundation SST, `METOFFICE-GLO-SST-L4-REP-OBS-SST` | 0.05°, daily |
| Degree Heating Weeks (reference) | [NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/product/5km/) DHW via ERDDAP | 0.05°, daily |

C3S is the default. It is the ESA Climate Change Initiative SST record, built to stay
consistent over decades, which matters because DHW compares today's SST with a 1980s
baseline taken from the same record. With NOAA's MMM method, C3S gives DHW closest
to NOAA Coral Reef Watch's (see Validation).

OSTIA is available as an option, but **its DHW runs well above NOAA's** and NOAA's
alert levels will over-warn with it. Relative to NOAA's CoralTemp, the OSTIA record is
cooler in the late 1980s and warmer in 2023, so its MMM comes out low and its recent
SST high, and both push DHW up. OSTIA is reasonable for marine heatwaves, whose
1991–2020 baseline is far less affected. Choose the product with `--sst` on the
command line, or `constants.SST_PRODUCT` in Python, and the CLI prints a warning when
OSTIA is used.

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
  (HotSpot and Degree Heating Week definitions) and
  [climatology](https://coralreefwatch.noaa.gov/product/5km/description_climatology.php)
  (MMM method)
- Whitaker & DeCarlo (2024) — Re(de)fining degree-heating week. *Coral Reefs*, 43, 969–984.
  DOI: [10.1007/s00338-024-02512-w](https://doi.org/10.1007/s00338-024-02512-w)
- Amaya et al. (2023) — Marine heatwaves need clear definitions so coastal communities
  can adapt. *Nature*, 616, 29–32

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
   the maximum monthly mean (MMM). HotSpots of at least 1 °C count in full, smaller
   ones count as zero, and DHW is their sum over a 12-week rolling window divided by 7.
   The MMM follows NOAA's method. For each pixel and calendar month, a straight-line
   trend is fitted through the 1985–2012 monthly means and evaluated at 1988.29, the
   centre of NOAA's original 1985–1990 + 1993 baseline. The MMM is the warmest of
   those 12 months. Keeping this older baseline makes reefugia's DHW comparable to
   NOAA's bleaching alert levels, which were calibrated on it, and avoids assuming
   corals adapt as fast as the ocean warms (Whitaker & DeCarlo 2024 found it the best
   predictor of observed bleaching). Setting `DHW_MMM_BASELINE = "wmo"` in
   `constants.py` uses the warmest 1991–2020 monthly mean instead

MHW and DHW deliberately use different baselines, because they answer different
questions. MHW detection asks how unusual the temperature is compared with a standard
recent climate, so it uses the WMO 1991–2020 normal. DHW asks how far the temperature
is above what corals were historically used to, so it uses NOAA's fixed late-1980s
baseline, on which NOAA's bleaching alert levels are calibrated. On the older baseline,
even more of 2023 would count as a marine heatwave, which says more about long-term
warming than about distinct heatwave events (Amaya et al. 2023).
Both come from one download (1985–2020) and are stored in one cache.

MHW and DHW use SST from before the start date as a lead-in, so the rolling windows
are full from the first day of the analysis period.

All intermediate products (climatology, MHW, DHW) are cached as Zarr stores in `data/`,
separately for each SST product, so subsequent runs are fast. A cache is recomputed automatically when the settings in
`reefugia/constants.py` or the CLI input dates or region change, but not when the code 
changes, so delete the relevant `data/*.zarr` after changing a calculation.

## Outputs

For each run, reefugia saves to `outputs/`:

**Data**
- `mhw_days_{region}_{sst}_{start}_{end}.nc` — spatial map of total MHW days
- `dhw_{region}_{sst}_{start}_{end}.nc` — daily DHW time series (time, lat, lon)
- `mhw_summary_{region}_{sst}_{start}_{end}.json` — MHW summary statistics
- `dhw_summary_{region}_{sst}_{start}_{end}.json` — DHW summary statistics
- `sst_anomaly_{region}_{sst}_{start}_{end}.csv` — monthly SST anomaly table

**Plots**
- `mhw_days_{region}_{sst}_{start}_{end}.png` — spatial map of MHW days with hotspot
- `dhw_{region}_{sst}_{start}_{end}.png` — spatial map of max DHW with hotspot
- `sst_anomaly_max_{region}_{sst}_{start}_{end}.png` — spatial map of the largest monthly
  SST anomaly relative to the 1991–2020 monthly mean
- `monthly_matrix_dhw_{region}_{sst}_{start}_{end}.png` — monthly DHW bar chart (single year) or matrix (multi-year)
- `dhw_animation_{region}_{sst}_{start}_{end}.mp4` — daily DHW animation

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
`dhw_vs_noaa_crw.ipynb` compares reefugia's 2023 DHW with
[NOAA Coral Reef Watch](https://coralreefwatch.noaa.gov/product/5km/) DHW across the
Red Sea.

With the defaults (C3S SST and NOAA's MMM method) the timing matches closely, with a
correlation of 0.995 for the regional mean through the year, and the level is close,
with a mean peak DHW of 11.9 against NOAA's 12.6 °C-weeks (bias −0.7, RMSE 2.8). The
spatial correlation of peak DHW is 0.68, because the two SST products differ locally,
most of all in the southern Red Sea.

Given NOAA's own SST and MMM, reefugia's DHW code reproduces NOAA's published DHW
across the whole Red Sea for all of 2023, with correlations of 1.000 in space and
time, zero mean bias and an RMSE of 0.003 °C-weeks (the rounding of NOAA's stored
values). The calculation matches NOAA's exactly, and every remaining difference comes
from the inputs.

Absolute DHW is very sensitive to small temperature offsets. A shift of 0.1 °C in SST
or MMM moves peak DHW by roughly 1.5–2 °C-weeks, so the choice of SST product and
baseline matters as much as the method. Every combination tested is below.

| SST | MMM | Spatial r | Mean peak DHW | Mean bias | RMSE |
|---|---|---|---|---|---|
| **C3S** | **NOAA method, from C3S (default)** | **0.68** | **11.9** | **−0.7** | **2.8** |
| C3S | warmest 1991–2020 monthly mean | 0.65 | 9.5 | −3.1 | 4.1 |
| OSTIA | NOAA method, from OSTIA | 0.66 | 21.6 | +9.0 | 9.9 |
| C3S | NOAA's own | 0.79 | 16.9 | +4.3 | 5.4 |
| OSTIA | NOAA's own | 0.82 | 17.2 | +4.6 | 5.5 |
| NOAA CoralTemp | NOAA's own | 1.000 | 12.6 | 0.0 | 0.003 |

NOAA's mean peak DHW is 12.6 °C-weeks. Bias and RMSE are in °C-weeks.

- Both Copernicus products run about 0.2 °C warmer than CoralTemp in 2023, so pairing
  either with NOAA's own MMM overshoots.
- Taking the MMM from the same product cancels that offset for C3S, which stays
  consistent with CoralTemp over the decades.
- It does not cancel for OSTIA. OSTIA's MMM is 0.3–0.5 °C below NOAA's while its 2023
  SST is 0.2 °C above, so its record has warmed more than CoralTemp's, and its DHW
  runs high. A plain average of NOAA's baseline years instead of the trend makes this
  slightly worse, so the drift is in the data, not the method.
- The earlier default (C3S with a 1991–2020 MMM) looked closer in the mean than its
  correlation suggests, because a warmer MMM and a warmer SST partly cancelled.

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
uv run reefugia --region red_sea --sst ostia   # OSTIA instead of the default C3S (DHW runs high)
```

`uv run` uses the project environment without activating it. With `.venv`
activated, plain `reefugia ...` (or `python -m reefugia ...`) works the same.