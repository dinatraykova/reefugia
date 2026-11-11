import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # no plot windows
import matplotlib.pyplot as plt
import click

from reefugia import constants, fetch, detect, analyse, visualise
from reefugia.constants import ANALYSIS_START, ANALYSIS_END

@click.command()
@click.option("--region", default="coral_triangle", show_default=True, help="Region to analyse")
@click.option("--start", default=ANALYSIS_START, show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=ANALYSIS_END, show_default=True, help="End date (YYYY-MM-DD)")
@click.option("--sst", "sst_product", type=click.Choice(sorted(fetch.SST_DATASETS)),
              default=constants.SST_PRODUCT, show_default=True,
              help="SST product: c3s (recommended for DHW) or ostia (DHW runs high, see README)")
def main(region, start, end, sst_product):
    """🪸 Reefugia — coral bleaching thermal stress analysis."""

    constants.SST_PRODUCT = sst_product
    tag = f"{region}_{sst_product}_{start}_{end}"

    start_time = time.time()
    if sst_product == "ostia":
        click.secho("⚠️  OSTIA SST gives DHW well above NOAA Coral Reef Watch's (about "
                    "+9 °C-weeks in the Red Sea in 2023), so NOAA's alert levels will "
                    "over-warn. Use --sst c3s for DHW; see the README.", fg="yellow")
    click.echo(f"\n🌊 Fetching {sst_product.upper()} SST for {region} from {start} to {end}...")
    ds = fetch.fetch_data(region, start, end)
    sst = ds["analysed_sst"]
    ocean_mask = ~sst.isel(time=0).isnull().compute()

    # MHW 
    mhw = detect.load_or_compute_mhw(sst, region, start, end)
    mhw_days = mhw["mhw"].sum(dim="time").compute()
    mhw_summary = analyse.summarise_mhw(mhw_days, ocean_mask)

    click.echo("\n── MHW Summary ──────────────────")
    click.echo(f"Mean MHW days:     {mhw_summary['mean_days']:.1f}")
    click.echo(f"Median MHW days:   {mhw_summary['median_days']:.1f}")
    click.echo(f"Max MHW days:      {mhw_summary['max_days']:.0f}")
    click.echo(f"% ocean affected:  {mhw_summary['pct_affected']:.1f}%")

    # DHW 
    dhw = detect.load_or_compute_dhw(sst, region, start, end)
    dhw_summary = analyse.summarise_dhw(dhw, ocean_mask)

    click.echo("\n── DHW Summary ──────────────────")
    click.echo(f"Mean peak DHW:  {dhw_summary['mean_peak_dhw']:.1f}")
    click.echo(f"Max DHW:        {dhw_summary['max_dhw']:.0f}")
    click.echo(f"% above 4 DHW:  {dhw_summary['pct_above_4']:.1f}%")
    click.echo(f"% above 8 DHW:  {dhw_summary['pct_above_8']:.1f}%")

    # Monthly SST anomaly 
    click.echo("\n── Monthly SST Anomaly ──────────")
    climatology = detect.load_or_compute_climatology(region)
    anomaly = analyse.monthly_sst_anomaly(sst, climatology["monthly_mean"], ocean_mask)
    click.echo(anomaly.to_string())

    # Save data 
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    anomaly.to_csv(os.path.join(output_dir, f"sst_anomaly_{tag}.csv"))

    with open(os.path.join(output_dir, f"mhw_summary_{tag}.json"), "w") as f:
        json.dump(mhw_summary, f, indent=2)

    with open(os.path.join(output_dir, f"dhw_summary_{tag}.json"), "w") as f:
        json.dump(dhw_summary, f, indent=2)

    mhw_days.to_netcdf(os.path.join(output_dir, f"mhw_days_{tag}.nc"))
    dhw.to_netcdf(os.path.join(output_dir, f"dhw_{tag}.nc"))

    click.echo(f"\nData saved to {output_dir}/")

    # Plots 
    click.echo("\n── Generating plots ─────────────")
    place = visualise.pretty_region(region)
    year = start[:4] if start[:4] == end[:4] else f"{start[:4]}–{end[:4]}"

    fig = visualise.plot_mhw_days(
        mhw_days,
        hotspot_lat=mhw_summary["hotspot_lat"],
        hotspot_lon=mhw_summary["hotspot_lon"],
        title=f"Marine heatwave days, {year} — {place}"
    )
    fig.savefig(os.path.join(output_dir, f"mhw_days_{tag}.png"),
                dpi=256, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    fig = visualise.plot_dhw(
        dhw,
        hotspot_lat=dhw_summary["hotspot_lat"],
        hotspot_lon=dhw_summary["hotspot_lon"],
        max_dhw=dhw_summary["max_dhw"],
        title=f"Peak Degree Heating Weeks, {year} — {place}"
    )
    fig.savefig(os.path.join(output_dir, f"dhw_{tag}.png"),
                dpi=256, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    fig = visualise.plot_sst_anomaly_max(
        sst,
        climatology,
        title=f"Largest monthly SST anomaly, {year} — {place}"
    )
    fig.savefig(os.path.join(output_dir, f"sst_anomaly_max_{tag}.png"),
                dpi=256, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    fig = visualise.plot_monthly_matrix(
        dhw,
        label="DHW (°C-weeks)",
        title=f"DHW by month, {year} — {place}"
    )
    fig.savefig(os.path.join(output_dir, f"monthly_matrix_dhw_{tag}.png"),
                dpi=256, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    click.echo(f"Plots saved to {output_dir}/")

    # Animation 
    click.echo("\n── Generating DHW animation ─────")
    anim = visualise.plot_dhw_animation(dhw, title=f"DHW — {place}")
    anim.save(
        os.path.join(output_dir, f"dhw_animation_{tag}.mp4"),
        writer="ffmpeg",
        fps=10,
        dpi = 80
    )
    plt.close()
    click.echo("Animation saved!")

    elapsed = time.time() - start_time
    click.echo(f"\n⏱️  Total run time: {int(elapsed // 60)}m {int(elapsed % 60)}s")