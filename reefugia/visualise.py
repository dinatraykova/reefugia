#TODO:
#      Interactive dashboard#

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import pandas as pd

def plot_mhw_days(mhw_days: xr.DataArray,
                  hotspot_lat: float,
                  hotspot_lon: float,
                  title: str = None) -> plt.Figure:

    fig, ax = plt.subplots(
        figsize=(8, 8), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)

    mesh = ax.pcolormesh(
        mhw_days.longitude,
        mhw_days.latitude,
        mhw_days.values,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        vmin=0,
        vmax=int(mhw_days.max().values)
    )
    plt.colorbar(mesh, ax=ax, label="Number of MHW days", pad=0.04)

    ax.plot(hotspot_lon, hotspot_lat,
            marker="*",
            color="blue",
            markersize=8,
            linestyle="none",
            transform=ccrs.PlateCarree(),
            label=f"Hotspot ({int(mhw_days.max().values)} days)")
    ax.legend()

    if title is None:
        title = "MHW days"
    ax.set_title(title)

    return fig

def plot_sst_anomaly_max(sst: xr.DataArray,
                         climatology: xr.Dataset,
                         title: str = None) -> plt.Figure:
    """Plot maximum SST anomaly at each pixel over the analysis period."""

    doy = sst.time.dt.dayofyear
    threshold = climatology["sst_threshold"].sel(dayofyear=doy, method="nearest")
    sst_anomaly = sst - threshold
    sst_anomaly_max = sst_anomaly.max(dim="time").compute()
    max_idx = sst_anomaly_max.argmax(dim=["latitude", "longitude"])
    hotspot_lat = float(sst_anomaly_max.latitude[max_idx["latitude"]].values)
    hotspot_lon = float(sst_anomaly_max.longitude[max_idx["longitude"]].values)

    anomaly_max_time_idx = int(sst_anomaly.max(dim=["latitude", "longitude"]).argmax(dim="time").values)
    anomaly_max_date = str(sst_anomaly.time[anomaly_max_time_idx].values)[:10]

    fig, ax = plt.subplots(
            figsize=(8, 8), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)

    mesh = ax.pcolormesh(
        sst_anomaly_max.longitude,
        sst_anomaly_max.latitude,
        sst_anomaly_max.values,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        vmin=0,
        vmax=int(sst_anomaly_max.max().values)
    )
    plt.colorbar(mesh, ax=ax, label="SST anomaly (°C)", pad=0.04)

    ax.plot(hotspot_lon, hotspot_lat,
            marker="*",
            color="blue",
            markersize=8,
            linestyle="none",
            transform=ccrs.PlateCarree(),
            label=f"Hotspot ({int(sst_anomaly_max.max().values)} °C) \nDate: {anomaly_max_date}"
    )
    ax.legend()
    plt.tight_layout()

    if title is None:
        title = "Maximum SST anomaly (°C)"
    ax.set_title(title)
    return fig

def plot_dhw(dhw: xr.DataArray,
             hotspot_lat: float,
             hotspot_lon: float,
             max_dhw: float,
             title: str = None) -> plt.Figure:

    dhw_max = dhw.max(dim="time").compute()
    dhw_max_time_idx = int(dhw.max(dim=["latitude", "longitude"]).argmax(dim="time").values)
    dhw_max_date = str(dhw.time[dhw_max_time_idx].values)[:10]
    #fig, ax, _ = _create_dhw_figure(dhw_max)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100,
                       subplot_kw={"projection": ccrs.PlateCarree()})
    _create_dhw_mesh(dhw_max, ax)

    ax.plot(hotspot_lon, hotspot_lat,
            marker="*", color="blue", markersize=10,
            linestyle="none", transform=ccrs.PlateCarree(),
            label=f"Max DHW ({max_dhw:.1f} °C-weeks) \nDate: {dhw_max_date}"
    )
    ax.legend()
    plt.tight_layout()

    if title is None:
        title = "Degree Heating Weeks (°C-weeks)"
    ax.set_title(title)
    return fig

def plot_dhw_animation(dhw: xr.DataArray,
                       title: str = None) -> animation.FuncAnimation:

    dhw_computed = dhw.compute()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100,
                           subplot_kw={"projection": ccrs.PlateCarree()})
#                           constrained_layout=True)
    mesh = _create_dhw_mesh(dhw_computed.isel(time=0), ax)
    date_title = ax.set_title(f"{title or 'DHW'} — {str(dhw.time[0].values)[:10]}")
    plt.tight_layout()

    def update(i):
        mesh.set_array(dhw_computed.isel(time=i).values.flatten())
        date_title.set_text(f"{title or 'DHW'} — {str(dhw.time[i].values)[:10]}")
        return mesh, date_title

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(dhw.time),
        interval=100,
        blit=True
    )
    return anim

def _create_dhw_mesh(dhw_2d: xr.DataArray, ax) -> object:
    """Set up mesh and colorbar for DHW plots.

    Args:
        dhw_2d: 2D DHW DataArray (latitude, longitude) for initial frame

    Returns:
        mesh
    """
    # vmax covers full NOAA bleaching alert scale (Alert 1-4: 4, 8, 12, 16°C-weeks)
    # scales up if data exceeds 16, capped at 20 (near complete mortality threshold)
    vmax = min(max(16, float(dhw_2d.max().values) * 1.1), 20)

    # custom colormap with bleaching threshold bands
    bounds = [0, 4, 8, 12, 16]
    colors = ["white", "yellow", "orange", "red", "darkred"]
    labels = [
            "0 — No stress",
            "4 — Alert 1\nbleaching likely",
            "8 — Alert 2\nsevere bleaching",
            "12 — Alert 3\nmass mortality",
            "16 — Alert 4\nnear complete mortality",
    ]

    if vmax > 16:
        bounds.append(vmax)
        colors.append("black")
        labels.append(f"{vmax:.0f}\nextreme mortality")

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "dhw", list(zip([b/vmax for b in bounds], colors)))

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)

    mesh = ax.pcolormesh(
        dhw_2d.longitude,
        dhw_2d.latitude,
        dhw_2d.values,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap,
        vmin=0,
        vmax=vmax
    )

    cbar = plt.colorbar(mesh, ax=ax, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(labels)

    return mesh
