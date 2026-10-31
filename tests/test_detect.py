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
