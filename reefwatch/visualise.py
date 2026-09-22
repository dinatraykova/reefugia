#TODO: 
#      Interactive dashboard#

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation

from reefwatch.constants import KELVIN_TO_CELSIUS

def plot_mhw_days(mhw_days: xr.DataArray,
                  hotspot_lat: float,
                  hotspot_lon: float,
                  title: str = None) -> plt.Figure:

    fig, ax = plt.subplots(
        figsize=(10, 6), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()})
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
    plt.colorbar(mesh, ax=ax, label="Number of MHW days", 
                 fraction=0.046, pad=0.04)

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

def plot_dhw(dhw: xr.DataArray,
             hotspot_lat: float,
             hotspot_lon: float,
             max_dhw: float,
             title: str = None) -> plt.Figure:

    dhw_max = dhw.max(dim="time").compute()
    fig, ax, _ = _create_dhw_figure(dhw_max)

    ax.plot(hotspot_lon, hotspot_lat,
            marker="*", color="blue", markersize=10,
            linestyle="none", transform=ccrs.PlateCarree(),
            label=f"Max DHW ({max_dhw:.1f} °C-weeks)")
    #ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=True)
    ax.legend()
    plt.tight_layout()

    if title is None:
        title = "Degree Heating Weeks (°C-weeks)"
    ax.set_title(title)
    return fig

def plot_dhw_animation(dhw: xr.DataArray,
                       title: str = None) -> animation.FuncAnimation:
    """Animate DHW progression through time.
    
    Args:
        dhw:   full DHW DataArray (time, latitude, longitude)
        title: optional title prefix
    """
    # setup figure once using first time step
    fig, ax, mesh = _create_dhw_figure(dhw.isel(time=0))
    date_title = ax.set_title(
        f"{title or 'DHW'} — {str(dhw.time[0].values)[:10]}")

    def update(i):
        mesh.set_array(dhw.isel(time=i).values.flatten())
        date_title.set_text(
            f"{title or 'DHW'} — {str(dhw.time[i].values)[:10]}")
        return mesh, date_title

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(dhw.time),
        interval=100,
        blit=True
    )

    return anim

def _create_dhw_figure(dhw_2d: xr.DataArray) -> tuple:
    """Set up figure, axes, mesh and colorbar for DHW plots.
    
    Args:
        dhw_2d: 2D DHW DataArray (latitude, longitude) for initial frame
    
    Returns:
        fig, ax, mesh
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

    fig, ax = plt.subplots(
        figsize=(10, 6), dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree()})
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

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(labels)

    return fig, ax, mesh