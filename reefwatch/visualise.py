#TODO: 
#      Time series plot (SST vs threshold) for a selected pixel
#      Animated map showing MHW/DHW progression through the year
#      Interactive dashboard#
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    #ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
    #      frameon=True)
    #plt.tight_layout()
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
    fig, ax = plt.subplots(
            figsize=(10, 6), dpi=100, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)

    # custom colormap with bleaching thresholds
    bounds = [0, 4, 8, 12, 20]
    colors = ["white", "yellow", "orange", "red", "darkred"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "dhw", list(zip([b/20 for b in bounds], colors)))

    mesh = ax.pcolormesh(
        dhw.longitude,
        dhw.latitude,
        dhw_max.values,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap,
        vmin=0,
        vmax=20
    )
    #plt.colorbar(mesh, ax=ax, label="DHW (°C-weeks)", shrink=0.8)
    plt.colorbar(mesh, ax=ax, label="Number of MHW days", 
             fraction=0.046, pad=0.04)

    ax.plot(hotspot_lon, hotspot_lat,
                marker="*",
                color="blue",
                markersize=8,
                linestyle="none",
                transform=ccrs.PlateCarree(),
                label=f"Hotspot({max_dhw:.0f} DHW)")
    #ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
    #          frameon=True)
    ax.legend()
    #plt.tight_layout()
    

    if title is None:
        title = "Degree Heating Weeks (°C-weeks)"
    ax.set_title(title)
    return fig