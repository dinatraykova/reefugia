"""Unit tests for MHW/DHW detection on small synthetic SST series (no network, no cache)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reefugia import detect, validate

THRESHOLD = 300.0  # K — used as both the MHW threshold and the MMM


def make_sst(values, start="2023-01-01"):
    """Single-pixel daily SST (K) with the given values."""
    values = np.asarray(values, dtype=float)
    return xr.DataArray(
        values[:, None, None],
        dims=("time", "latitude", "longitude"),
        coords={"time": pd.date_range(start, periods=len(values), freq="D"),
                "latitude": [0.025], "longitude": [0.025]},
    )


@pytest.fixture(autouse=True)
def flat_climatology(monkeypatch):
    # flat threshold/MMM, and a lead-in that sits 1 °C below it
    climatology = {
        "sst_threshold": xr.DataArray(np.full(366, THRESHOLD), dims="dayofyear",
                                      coords={"dayofyear": np.arange(1, 367)}),
        "mmm": xr.DataArray(THRESHOLD),
    }
    monkeypatch.setattr(detect, "load_or_compute_climatology", lambda region: climatology)

    def fake_fetch(region, start, end):
        n = len(pd.date_range(start, end, freq="D"))
        return xr.Dataset({"analysed_sst": make_sst(np.full(n, THRESHOLD - 1), start=start)})
    monkeypatch.setattr(detect.fetch, "fetch_data", fake_fetch)


def mhw_flags(values):
    return detect.compute_mhw(make_sst(values), "test")["mhw"].values.ravel()


def test_four_hot_days_is_not_a_heatwave():
    flags = mhw_flags([299] * 5 + [301] * 4 + [299] * 5)
    assert not flags.any()


def test_five_hot_days_flags_all_five():
    flags = mhw_flags([299] * 5 + [301] * 5 + [299] * 5)
    assert flags.tolist() == [False] * 5 + [True] * 5 + [False] * 5


def test_events_two_days_apart_are_joined():
    flags = mhw_flags([299] * 3 + [301] * 5 + [299] * 2 + [301] * 5 + [299] * 3)
    assert flags.tolist() == [False] * 3 + [True] * 12 + [False] * 3


def test_events_three_days_apart_stay_separate():
    flags = mhw_flags([299] * 3 + [301] * 5 + [299] * 3 + [301] * 5 + [299] * 3)
    assert flags.tolist() == [False] * 3 + [True] * 5 + [False] * 3 + [True] * 5 + [False] * 3


def test_short_run_is_not_joined_to_an_event():
    # 3 hot days after a 1-day gap are not an event, so nothing is joined
    flags = mhw_flags([301] * 5 + [299] + [301] * 3 + [299] * 3)
    assert flags.tolist() == [True] * 5 + [False] * 7


def test_event_before_the_start_date_is_joined_across_the_gap(monkeypatch):
    # lead-in: an event ending two days before the start, then a cool day
    def fetch_hot_lead_in(region, start, end):
        n = len(pd.date_range(start, end, freq="D"))
        lead = [THRESHOLD + 1] * (n - 1) + [THRESHOLD - 1]
        return xr.Dataset({"analysed_sst": make_sst(lead, start=start)})
    monkeypatch.setattr(detect.fetch, "fetch_data", fetch_hot_lead_in)

    flags = mhw_flags([299] + [301] * 5 + [299] * 3)
    assert flags.tolist() == [True] * 6 + [False] * 3


def hobday_reference(above, min_duration=5, max_gap=2):
    """Plain-loop Hobday et al. (2016) event logic: runs of >= min_duration days
    above threshold, joined across gaps of <= max_gap days."""
    events, start = [], None
    for t, hot in enumerate(list(above) + [False]):
        if hot and start is None:
            start = t
        elif not hot and start is not None:
            if t - start >= min_duration:
                events.append([start, t - 1])
            start = None
    joined = []
    for ev in events:
        if joined and ev[0] - joined[-1][1] - 1 <= max_gap:
            joined[-1][1] = ev[1]
        else:
            joined.append(ev)
    flags = np.zeros(len(above), dtype=bool)
    for s, e in joined:
        flags[s:e + 1] = True
    return flags


@pytest.mark.parametrize("seed", range(20))
def test_mhw_matches_loop_reference_on_random_series(seed):
    rng = np.random.default_rng(seed)
    above = rng.random(365) < rng.uniform(0.5, 0.9)  # long and short runs and gaps
    flags = mhw_flags(np.where(above, THRESHOLD + 1, THRESHOLD - 1))
    np.testing.assert_array_equal(flags, hobday_reference(above))


def test_constant_2C_hotspot_for_84_days_gives_dhw_24():
    dhw = detect.compute_dhw(make_sst([THRESHOLD + 2] * 84), "test")
    assert dhw.sizes["time"] == 84  # lead-in trimmed off
    assert dhw.values.ravel()[-1] == pytest.approx(24.0)


def test_hotspot_below_1C_gives_dhw_0():
    dhw = detect.compute_dhw(make_sst([THRESHOLD + 0.5] * 84), "test")
    assert np.all(dhw.values == 0)


def test_compare_dhw_identical_after_regridding():
    # reference stored north → south and stamped at 12:00, like NOAA CRW on ERDDAP
    t = pd.date_range("2023-01-01", periods=30, freq="D")
    lat, lon = [0.025, 0.075, 0.125], [0.025, 0.075]
    scale = np.arange(6).reshape(3, 2) + 1.0
    ours = xr.DataArray(np.linspace(0, 1, 30)[:, None, None] * scale,
                        dims=("time", "latitude", "longitude"),
                        coords={"time": t, "latitude": lat, "longitude": lon})
    ref = ours.isel(latitude=slice(None, None, -1)).assign_coords(time=t + pd.Timedelta(hours=12))
    ocean = xr.ones_like(ours.isel(time=0), dtype=bool)

    result = validate.compare_dhw(ours, ref - 1, ocean)
    assert result["n_pixels"] == 6
    assert result["r_spatial"] == pytest.approx(1.0)
    assert result["mean_bias"] == pytest.approx(1.0)


# ── Climatology threshold (Hobday et al. 2016) ────────────────────────────────

def test_day_of_year_uses_a_leap_year_calendar():
    dates = pd.to_datetime(["2023-02-28", "2023-03-01", "2024-02-29",
                            "2024-03-01", "2023-12-31", "2024-12-31"])
    assert detect.day_of_year(dates).tolist() == [59, 61, 60, 61, 366, 366]


def threshold_reference(values, dates, pct=90, half_width=5, smoothing=31):
    """Plain-loop version of the Hobday et al. (2016) threshold for one series."""
    doy = detect.day_of_year(dates)
    raw = np.full(366, np.nan)
    for d in range(1, 367):
        if d == 60:
            continue
        pooled = [values[c + w] for c in np.flatnonzero(doy == d)
                  for w in range(-half_width, half_width + 1) if 0 <= c + w < len(values)]
        raw[d - 1] = np.percentile(pooled, pct)
    raw[59] = (raw[58] + raw[60]) / 2
    h = smoothing // 2
    return np.array([np.mean([raw[(d + k) % 366] for k in range(-h, h + 1)]) for d in range(366)])


def baseline_sst(years=6, seed=0):
    dates = pd.date_range("2000-01-01", periods=365 * years + years // 4, freq="D")
    rng = np.random.default_rng(seed)
    season = 3 * np.sin(2 * np.pi * (dates.day_of_year.to_numpy() - 100) / 365.25)
    values = 300 + season + rng.normal(0, 0.5, len(dates))
    return values, make_sst(values, start="2000-01-01")


def test_threshold_matches_loop_reference():
    values, sst = baseline_sst()
    ours = detect.compute_threshold(sst).values.ravel()
    np.testing.assert_allclose(ours, threshold_reference(values, sst.time.values), rtol=0, atol=1e-9)


def test_threshold_wraps_smoothly_from_december_to_january():
    _, sst = baseline_sst(years=10)
    threshold = detect.compute_threshold(sst).values.ravel()
    steps = np.abs(np.diff(threshold))
    assert abs(threshold[0] - threshold[-1]) <= 2 * steps.max()


def test_constant_sst_gives_constant_threshold():
    sst = make_sst(np.full(365 * 3 + 1, 301.5), start="2000-01-01")
    np.testing.assert_allclose(detect.compute_threshold(sst).values, 301.5)


# ── Validation helpers ────────────────────────────────────────────────────────

def test_mmm_from_crw_recovers_sst_minus_hotspot():
    t = pd.date_range("2023-01-01", periods=6, freq="D")
    mmm = 29.5
    sst = xr.DataArray([28.0, 29.0, 30.0, 31.0, 29.5, 30.5], dims="time", coords={"time": t})
    hotspot = sst - mmm  # 29.5 gives a HotSpot of exactly 0, which is skipped
    assert float(validate.mmm_from_crw(sst, hotspot)) == pytest.approx(mmm)


def test_compute_dhw_accepts_another_mmm():
    # SST 2 °C above the default MMM, but only 0.5 °C above the one passed in
    sst = make_sst([THRESHOLD + 2] * 84)
    dhw = detect.compute_dhw(sst, "test", mmm=xr.DataArray(THRESHOLD + 1.5))
    assert np.all(dhw.values == 0)


def test_compute_dhw_without_lead_in_uses_only_the_given_sst(monkeypatch):
    def no_fetch(*args):
        raise AssertionError("lead_in=False should not fetch")
    monkeypatch.setattr(detect.fetch, "fetch_data", no_fetch)

    dhw = detect.compute_dhw(make_sst([THRESHOLD + 2] * 84), "test", lead_in=False)
    values = dhw.values.ravel()
    assert values[0] == pytest.approx(2 / 7)   # window holds one day
    assert values[-1] == pytest.approx(24.0)   # window full on day 84
