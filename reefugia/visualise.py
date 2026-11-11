import functools

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio_ffmpeg

from reefugia.analyse import hotspot, TOP_QUANTILE

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# ── Style ─────────────────────────────────────────────────────────────────────

INK = "#0b0b0b"            # titles, values
INK_SECONDARY = "#52514e"  # axis labels, ticks
INK_MUTED = "#8a8984"      # reference-line labels
GRID = "#e4e3df"
LAND = "#d9d7d1"
COAST = "#7a7975"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

_STYLE = {
    "font.size": 10,
    "text.color": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

# sequential warm ramp (light → dark) for magnitudes: MHW days, SST anomaly
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "reefugia_heat", ["#fdf1ec", "#f7c2ac", "#ec835a", "#c8432a", "#7a1d0c"])

# diverging blue ↔ red with a neutral grey midpoint, for signed differences/trends
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "reefugia_diverging",
    ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#8f1d1c"])

# NOAA Coral Reef Watch bleaching alert levels, as discrete DHW bands
DHW_BOUNDS = [0, 4, 8, 12, 16, 20, 24]
DHW_COLORS = ["#fbf7ec", "#f6c445", "#ec835a", "#d03b3b", "#8f1d1c", "#3a0b0b"]
DHW_LABELS = ["No alert  (< 4)", "Alert 1  (4–8)", "Alert 2  (8–12)",
              "Alert 3  (12–16)", "Alert 4  (16–20)", "Alert 5  (≥ 20)"]
DHW_CMAP = mcolors.ListedColormap(DHW_COLORS, name="reefugia_dhw")
DHW_NORM = mcolors.BoundaryNorm(DHW_BOUNDS, DHW_CMAP.N, clip=True)


def _styled(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(_STYLE):
            return func(*args, **kwargs)
    return wrapper


def pretty_region(region_name: str) -> str:
    """'great_barrier_reef' → 'Great Barrier Reef'."""
    return region_name.replace("_", " ").title()


# ── Shared map pieces ─────────────────────────────────────────────────────────

def _map_axes(da: xr.DataArray, size: float = 8.0):
    lon, lat = da.longitude.values, da.latitude.values
    half = abs(float(lon[1] - lon[0])) / 2 if lon.size > 1 else 0.025
    extent = [lon.min() - half, lon.max() + half, lat.min() - half, lat.max() + half]

    # size the figure to the region's shape, leaving room for the colourbar
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    if aspect >= 1:
        figsize = (size / aspect + 2.6, size)
    else:
        figsize = (size + 2.6, size * aspect + 1.2)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained",
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=LAND, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor=COAST, linewidth=0.5, zorder=3)
    ax.spines["geo"].set_edgecolor("#c3c2b7")

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#d6d5d0",
                      linestyle="--", zorder=4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": INK_SECONDARY}
    return fig, ax


def _colorbar(fig, mappable, ax, label: str, **kwargs):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.85, aspect=30, pad=0.03, **kwargs)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK_SECONDARY, labelsize=9)
    cb.set_label(label, color=INK_SECONDARY)
    return cb


def _mark_hotspot(ax, lon: float, lat: float, label: str):
    ax.plot(lon, lat, marker="*", markersize=15, markerfacecolor="white",
            markeredgecolor=INK, markeredgewidth=1.0, linestyle="none",
            transform=ccrs.PlateCarree(), zorder=5, label=label)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="none", borderpad=0.6, handletextpad=0.4)


def _dhw_mesh(fig, dhw_2d: xr.DataArray, ax):
    mesh = ax.pcolormesh(dhw_2d.longitude, dhw_2d.latitude, dhw_2d.values,
                         cmap=DHW_CMAP, norm=DHW_NORM, shading="nearest",
                         transform=ccrs.PlateCarree(), zorder=1)
    centres = [(lo + hi) / 2 for lo, hi in zip(DHW_BOUNDS[:-1], DHW_BOUNDS[1:])]
    cb = _colorbar(fig, mesh, ax, "NOAA bleaching alert level (DHW, °C-weeks)",
                   ticks=centres)
    cb.set_ticklabels(DHW_LABELS)
    return mesh


# ── Maps ──────────────────────────────────────────────────────────────────────

@_styled
def plot_mhw_days(mhw_days: xr.DataArray,
                  hotspot_lat: float,
                  hotspot_lon: float,
                  p99_days: float,
                  title: str = None) -> plt.Figure:
    """Map of MHW days per pixel, with the hotspot marked."""
    top = p99_days
    fig, ax = _map_axes(mhw_days)
    mesh = ax.pcolormesh(mhw_days.longitude, mhw_days.latitude, mhw_days.values,
                         cmap=HEAT_CMAP, vmin=0, vmax=top,
                         shading="nearest", transform=ccrs.PlateCarree(), zorder=1)
    _colorbar(fig, mesh, ax, "Marine heatwave days", extend="max")
    _mark_hotspot(ax, hotspot_lon, hotspot_lat, f"Hotspot (top 1%, {top:.0f}+ days)")
    ax.set_title(title or "Marine heatwave days")
    return fig


@_styled
def plot_sst_anomaly_max(sst: xr.DataArray,
                         climatology: xr.Dataset,
                         title: str = None) -> plt.Figure:
    """Map of max monthly SST anomaly (vs 1991–2020 monthly mean) per pixel."""
    sst_monthly = sst.resample(time="MS").mean()
    clim = climatology["monthly_mean"].sel(month=sst_monthly.time.dt.month).drop_vars("month")
    sst_anomaly = (sst_monthly - clim).compute()
    sst_anomaly_max = sst_anomaly.max(dim="time")

    top = float(sst_anomaly_max.quantile(TOP_QUANTILE))
    hotspot_lat, hotspot_lon = hotspot(sst_anomaly_max)
    at_hotspot = sst_anomaly.sel(latitude=hotspot_lat, longitude=hotspot_lon)
    hotspot_month = str(at_hotspot.idxmax(dim="time").values)[:7]

    fig, ax = _map_axes(sst_anomaly_max)
    mesh = ax.pcolormesh(sst_anomaly_max.longitude, sst_anomaly_max.latitude,
                         sst_anomaly_max.values, cmap=HEAT_CMAP, vmin=0,
                         vmax=top, shading="nearest",
                         transform=ccrs.PlateCarree(), zorder=1)
    _colorbar(fig, mesh, ax, "Largest monthly SST anomaly (°C)", extend="max")
    _mark_hotspot(ax, hotspot_lon, hotspot_lat,
                  f"Hotspot (top 1%, {top:.1f}+ °C, {hotspot_month})")
    ax.set_title(title or "Largest monthly SST anomaly")
    return fig


@_styled
def plot_dhw(dhw: xr.DataArray,
             hotspot_lat: float,
             hotspot_lon: float,
             p99_dhw: float,
             title: str = None) -> plt.Figure:
    """Map of peak DHW per pixel with NOAA alert bands, hotspot marked."""
    dhw_max = dhw.max(dim="time").compute()
    at_hotspot = dhw.sel(latitude=hotspot_lat, longitude=hotspot_lon)
    peak_date = str(at_hotspot.idxmax(dim="time").values)[:10]

    fig, ax = _map_axes(dhw_max)
    _dhw_mesh(fig, dhw_max, ax)
    _mark_hotspot(ax, hotspot_lon, hotspot_lat,
                  f"Hotspot (top 1%, {p99_dhw:.1f}+ °C-weeks, {peak_date})")
    ax.set_title(title or "Peak Degree Heating Weeks")
    return fig


@_styled
def plot_dhw_animation(dhw: xr.DataArray,
                       title: str = None) -> animation.FuncAnimation:
    """Animation of daily DHW with NOAA alert bands."""
    dhw_computed = dhw.compute()
    dates = [str(t)[:10] for t in dhw_computed.time.values]
    name = title or "Degree Heating Weeks"

    fig, ax = _map_axes(dhw_computed.isel(time=0))
    mesh = _dhw_mesh(fig, dhw_computed.isel(time=0), ax)
    date_title = ax.set_title(f"{name} — {dates[0]}")

    def update(i):
        mesh.set_array(dhw_computed.isel(time=i).values.ravel())
        date_title.set_text(f"{name} — {dates[i]}")
        return mesh, date_title

    return animation.FuncAnimation(fig, update, frames=len(dates), interval=100, blit=True)


@_styled
def plot_trend_map(slope_da: xr.DataArray,
                   title: str = "DHW trend") -> plt.Figure:
    """Map of per-pixel DHW trend from pixel_trend()."""
    vmax = float(np.nanpercentile(np.abs(slope_da.values), 97))
    fig, ax = _map_axes(slope_da)
    mesh = ax.pcolormesh(slope_da.longitude, slope_da.latitude, slope_da.values,
                         cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, shading="nearest",
                         transform=ccrs.PlateCarree(), zorder=1)
    _colorbar(fig, mesh, ax, "DHW trend (°C-weeks per year)", extend="both")
    ax.set_title(title)
    return fig


@_styled
def plot_dhw_difference(diff: xr.DataArray,
                        r_spatial: float,
                        title: str = None) -> plt.Figure:
    """Map of peak DHW minus reference peak DHW (e.g. NOAA CRW), from compare_dhw()."""
    vmax = float(np.nanpercentile(np.abs(diff.values), 98))
    fig, ax = _map_axes(diff)
    mesh = ax.pcolormesh(diff.longitude, diff.latitude, diff.values,
                         cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, shading="nearest",
                         transform=ccrs.PlateCarree(), zorder=1)
    _colorbar(fig, mesh, ax, "Peak DHW difference (°C-weeks)", extend="both")
    ax.text(0.97, 0.97, f"spatial r = {r_spatial:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, zorder=6,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=4))
    ax.set_title(title or "Peak DHW: reefugia − NOAA CRW")
    return fig


@_styled
def plot_map(da: xr.DataArray,
             label: str,
             title: str = None,
             cmap=HEAT_CMAP,
             **kwargs) -> plt.Figure:
    """Map of any 2D field (latitude, longitude) in the reefugia style.
    Extra keyword arguments (e.g. vmin, vmax) go to pcolormesh."""
    fig, ax = _map_axes(da)
    mesh = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap=cmap,
                         shading="nearest", transform=ccrs.PlateCarree(), zorder=1, **kwargs)
    _colorbar(fig, mesh, ax, label)
    if title:
        ax.set_title(title)
    return fig


# ── Charts ────────────────────────────────────────────────────────────────────

@_styled
def plot_monthly_matrix(da: xr.DataArray,
                        label: str = "DHW (°C-weeks)",
                        title: str = None,
                        alert_levels: bool = True) -> plt.Figure:
    """Month × year matrix of regional mean and top 1% of pixels (bar chart for one year)."""
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def by_month_year(series, how):
        df = series.compute().to_series().rename("value").reset_index()
        df["year"], df["month"] = df["time"].dt.year, df["time"].dt.month
        return df.groupby(["month", "year"])["value"].agg(how).unstack("year")

    matrix = by_month_year(da.mean(dim=["latitude", "longitude"]), "mean")
    if da.chunks:
        da = da.chunk({"latitude": -1, "longitude": -1})
    top_pixels = da.quantile(TOP_QUANTILE, dim=["latitude", "longitude"]).drop_vars("quantile")
    matrix_max = by_month_year(top_pixels, "max")

    if len(matrix.columns) == 1:
        year = matrix.columns[0]
        mean = matrix[year].reindex(range(1, 13)).values
        peak = matrix_max[year].reindex(range(1, 13)).values
        y = np.arange(12)

        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        ax.hlines(y, mean, peak, color="#c3c2b7", linewidth=1, zorder=1)
        ax.barh(y, mean, height=0.62, color=ORANGE, zorder=2, label="Regional mean")
        ax.plot(peak, y, "o", markersize=7, markerfacecolor=INK, markeredgecolor="white",
                markeredgewidth=1.2, linestyle="none", zorder=3, label="Top 1% of pixels")

        if alert_levels:
            # grid lines sit on the alert thresholds, labelled at the top
            top = max(DHW_BOUNDS[1], float(np.nanmax(peak)))
            shown = [b for b in DHW_BOUNDS if b <= top * 1.1]
            ax.set_xticks(shown)
            for value, name in zip(shown[1:], DHW_LABELS[1:]):
                ax.text(value, -0.8, name.split("  ")[0], ha="center", va="bottom",
                        fontsize=8, color=INK_MUTED)

        ax.set_yticks(y, month_labels)
        ax.set_ylim(11.6, -1.2)  # January at the top
        ax.set_xlim(left=0)
        ax.set_xlabel(label)
        ax.xaxis.grid(True, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower right")
        ax.set_title(title or f"Monthly DHW — {year}")

    else:
        years = list(matrix.columns)
        vmax = float(np.nanmax(matrix_max.values))
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.5 * len(years) + 6), 5.5),
                                 layout="constrained", sharey=True)
        for ax, mat, subtitle in zip(axes, [matrix, matrix_max],
                                     ["Regional mean", "Top 1% of pixels"]):
            mesh = ax.pcolormesh(np.arange(len(years) + 1), np.arange(13),
                                 mat.reindex(range(1, 13)).values,
                                 cmap=HEAT_CMAP, vmin=0, vmax=vmax,
                                 edgecolors="white", linewidth=1.5)
            ax.set_xticks(np.arange(len(years)) + 0.5, years,
                          rotation=45 if len(years) > 8 else 0)
            ax.set_yticks(np.arange(12) + 0.5, month_labels)
            ax.set_ylim(12, 0)
            ax.tick_params(length=0)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(subtitle, fontsize=11)
        _colorbar(fig, mesh, axes, label)
        fig.suptitle(title or f"Monthly {label}", x=0.01, ha="left",
                     fontsize=13, fontweight="bold")

    return fig


@_styled
def plot_trend(trend: dict,
               region_name: str = "",
               title: str = None) -> plt.Figure:
    """Annual max DHW with fitted trend."""
    years, values, y_fit = trend["years"], trend["values"], trend["y_fit"]
    degree_label = {1: "Linear", 2: "Quadratic", 3: "Cubic"}.get(
        trend["degree"], f"Degree-{trend['degree']}")

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.axhline(4, color="#b5b4ae", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(years[0] - 0.6, 4.1, "Alert 1 (4 °C-weeks)", ha="left", va="bottom",
            fontsize=8, color=INK_MUTED)

    ax.plot(years, values, "o", markersize=6, markerfacecolor=BLUE,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3,
            label="Annual max DHW (regional mean)")
    ax.plot(years, y_fit, color=ORANGE, linewidth=2, zorder=2,
            label=f"{degree_label} fit (R² = {trend['r2']:.2f})")

    ax.set_xlabel("Year")
    ax.set_ylabel("DHW (°C-weeks)")
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper left")
    ax.set_title(title or f"Annual max DHW — {pretty_region(region_name)}")
    return fig
