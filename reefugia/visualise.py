import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import imageio_ffmpeg
import matplotlib as mpl
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def plot_mhw_days(mhw_days: xr.DataArray,
                  hotspot_lat: float,
                  hotspot_lon: float,
                  title: str = None) -> plt.Figure:
    """Map of MHW days per pixel, with the hotspot marked."""
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
    ax.legend(loc="upper right")

    if title is None:
        title = "MHW days"
    ax.set_title(title)

    return fig

def plot_sst_anomaly_max(sst: xr.DataArray,
                         climatology: xr.Dataset,
                         title: str = None) -> plt.Figure:
    """Map of max monthly SST anomaly (vs 1991–2020 monthly mean) per pixel."""
    sst_monthly = sst.resample(time="MS").mean()
    clim = climatology["monthly_mean"].sel(month=sst_monthly.time.dt.month).drop_vars("month")
    sst_anomaly = (sst_monthly - clim).compute()
    sst_anomaly_max = sst_anomaly.max(dim="time")

    max_idx = sst_anomaly_max.argmax(dim=["latitude", "longitude"])
    hotspot_lat = float(sst_anomaly_max.latitude[max_idx["latitude"]].values)
    hotspot_lon = float(sst_anomaly_max.longitude[max_idx["longitude"]].values)
    hotspot_month = str(sst_anomaly.isel(max_idx).idxmax(dim="time").values)[:7]

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
        vmax=float(sst_anomaly_max.max().values)
    )
    plt.colorbar(mesh, ax=ax, label="Monthly SST anomaly (°C)", pad=0.04)

    ax.plot(hotspot_lon, hotspot_lat,
            marker="*",
            color="blue",
            markersize=8,
            linestyle="none",
            transform=ccrs.PlateCarree(),
            label=f"Hotspot ({float(sst_anomaly_max.max().values):.1f} °C) \nMonth: {hotspot_month}"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()

    if title is None:
        title = "Max monthly SST anomaly (°C)"
    ax.set_title(title)
    return fig

def plot_dhw(dhw: xr.DataArray,
             hotspot_lat: float,
             hotspot_lon: float,
             max_dhw: float,
             title: str = None) -> plt.Figure:
    """Map of peak DHW per pixel with NOAA alert bands, hotspot marked."""
    dhw_max = dhw.max(dim="time").compute()
    dhw_max_time_idx = int(dhw.max(dim=["latitude", "longitude"]).argmax(dim="time").values)
    dhw_max_date = str(dhw.time[dhw_max_time_idx].values)[:10]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100,
                       subplot_kw={"projection": ccrs.PlateCarree()})
    _create_dhw_mesh(dhw_max, ax, max_dhw)

    ax.plot(hotspot_lon, hotspot_lat,
            marker="*", color="blue", markersize=10,
            linestyle="none", transform=ccrs.PlateCarree(),
            label=f"Max DHW ({max_dhw:.1f} °C-weeks) \nDate: {dhw_max_date}"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()

    if title is None:
        title = "Degree Heating Weeks (°C-weeks)"
    ax.set_title(title)
    return fig

def plot_dhw_animation(dhw: xr.DataArray,
                       max_dhw: float,
                       title: str = None) -> animation.FuncAnimation:

    dhw_computed = dhw.compute()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100,
                           subplot_kw={"projection": ccrs.PlateCarree()})
    mesh = _create_dhw_mesh(dhw_computed.isel(time=0), ax, max_dhw)
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

def plot_monthly_matrix(da: xr.DataArray,
                        label: str = "DHW (°C-weeks)",
                        title: str = None) -> plt.Figure:
    """Month × year matrix of regional means (bar chart for one year)."""
    da_mean = da.mean(dim=["latitude", "longitude"]).compute()
    df = da_mean.to_series().rename("value").reset_index()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    matrix = df.groupby(["month", "year"])["value"].mean().reset_index()
    matrix = matrix.pivot(index="month", columns="year", values="value")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    da_max = da.max(dim=["latitude", "longitude"]).compute()
    df_max = da_max.to_series().rename("value").reset_index()
    df_max["year"] = df_max["time"].dt.year
    df_max["month"] = df_max["time"].dt.month
    matrix_max = df_max.groupby(["month", "year"])["value"].max().reset_index()
    matrix_max = matrix_max.pivot(index="month", columns="year", values="value")

    if len(matrix.columns) == 1:
        year = matrix.columns[0]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.barh(range(12), matrix[year].values,
            color="#b03a2e", alpha=0.5, label="Spatial mean")
        ax.barh(range(12), matrix_max[year].values,
            color="none", edgecolor="#7b241c", linewidth=1.5,
            linestyle="--", label="Spatial max")
        ax.set_yticks(range(12))
        ax.set_yticklabels(month_labels)
        ax.set_xlabel(label)
        ax.set_title(title or f"Monthly DHW — {year}")
        ax.legend(loc="upper right")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=100)
        matrix_max = df_max.groupby(["month", "year"])["value"].max().reset_index()
        matrix_max = matrix_max.pivot(index="month", columns="year", values="value")

        for ax, mat, subtitle in zip(axes, [matrix, matrix_max], ["Spatial mean", "Spatial max"]):
            im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", origin="lower")
            plt.colorbar(im, ax=ax, label=label, pad=0.02)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns)
            ax.set_yticks(range(12))
            ax.set_yticklabels(month_labels)
            ax.set_xlabel("Year")
            ax.set_title(f"{title or f'Monthly {label}'} — {subtitle}")

    plt.tight_layout()
    return fig

def plot_trend(trend: dict,
               region_name: str = "",
               title: str = None) -> plt.Figure:
    """Annual max DHW with fitted trend."""
    years  = trend["years"]
    values = trend["values"]
    y_fit  = trend["y_fit"]
    r2     = trend["r2"]
    degree = trend["degree"]
    degree_label = {1: "linear", 2: "quadratic", 3: "cubic"}.get(degree, f"degree-{degree}")

    fig, ax = plt.subplots(figsize=(11, 5), dpi=110)

    # observed points
    ax.scatter(years, values, color="#1a6b9a", s=28, zorder=3, label="Annual max DHW")

    # fitted curve
    ax.plot(years, y_fit, color="#c0392b", linewidth=2,
            label=f"{degree_label} fit  (R²={r2:.2f})")

    # bleaching threshold reference
    ax.axhline(4, color="#e67e22", linewidth=1, linestyle="--", alpha=0.7,
               label="Bleaching threshold (4 °C-weeks)")

    ax.set_xlabel("Year")
    ax.set_ylabel("DHW (°C-weeks)")
    ax.set_title(title or f"Annual max DHW trend — {region_name}")
    ax.legend(fontsize=9, framealpha=0.7, loc="upper right")
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_trend_map(slope_da: xr.DataArray,
                   title: str = "DHW trend (°C-weeks yr⁻¹)",
                   land_color: str = "#c8c8c8") -> plt.Figure:
    """Map of per-pixel DHW trend from pixel_trend()."""
    lats = slope_da.latitude.values
    lons = slope_da.longitude.values
    data = slope_da.values

    # symmetric colour scale around zero
    vmax = np.nanpercentile(np.abs(data), 97)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 6), dpi=110,
                           subplot_kw={"projection": ccrs.PlateCarree()})

    im = ax.pcolormesh(lons, lats, data,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       shading="nearest", transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.03)
    cbar.set_label("DHW trend (°C-weeks yr⁻¹)")

    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_dhw_difference(diff: xr.DataArray,
                        r_spatial: float,
                        title: str = None,
                        land_color: str = "#c8c8c8") -> plt.Figure:
    """Map of peak DHW minus reference peak DHW (e.g. NOAA CRW), from compare_dhw()."""
    vmax = float(np.nanpercentile(np.abs(diff.values), 98))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100,
                           subplot_kw={"projection": ccrs.PlateCarree()})
    mesh = ax.pcolormesh(diff.longitude, diff.latitude, diff.values,
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         shading="nearest", transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    plt.colorbar(mesh, ax=ax, label="Peak DHW difference (°C-weeks)", pad=0.04)

    ax.text(0.97, 0.97, f"spatial r = {r_spatial:.2f}", transform=ax.transAxes,
            ha="right", va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_title(title or "Peak DHW: reefugia − NOAA CRW")
    plt.tight_layout()
    return fig

def _create_dhw_mesh(dhw_2d: xr.DataArray, ax, max_dhw: float) -> object:
    # vmax covers full NOAA bleaching alert scale (Alert 1-4: 4, 8, 12, 16°C-weeks)
    # scales up if data exceeds 16, capped at 20 (near complete mortality threshold)
    vmax = min(max(16, max_dhw * 1.1), 20)

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
