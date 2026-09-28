import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import click

# Local 
from reefwatch import fetch, detect, analyse, visualise
from reefwatch.constants import ANALYSIS_START, ANALYSIS_END

@click.command()
# Region for analysis, default is coral_triangle
@click.option("--region", default="coral_triangle", show_default=True, help="Region to analyse")
@click.option("--start", default=ANALYSIS_START, show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=ANALYSIS_END, show_default=True, help="End date (YYYY-MM-DD)")
#@click.option("--comparison_years", default="1991,2023", show_default=True, 
#              help="Comma-separated years for MHW comparison (e.g. 1991,2023)")

def main(region, start, end):
    """🪸 Reefwatch — coral bleaching thermal stress analysis."""

    start_time = time.time()
    click.echo(f"\n🌊 Fetching SST data for {region} from {start} to {end}...")
    
    ds = fetch.fetch_data(region,start,end)

    sst = ds["analysed_sst"]
    ocean_mask = ~sst.isel(time=0).isnull().compute()  # True where ocean, False where land

    mhw = detect.load_or_compute_mhw(sst, region, start, end)
    mhw_days = mhw["mhw"].sum(dim="time").compute()
    mhw_summary = analyse.summarise_mhw(mhw_days, ocean_mask)
    click.echo("\n── MHW Summary ──────────────────")
    click.echo(f"Mean MHW days: {mhw_summary['mean_days']:.1f}")
    click.echo(f"Median MHW days: {mhw_summary['median_days']:.1f}")
    click.echo(f"Max MHW days: {mhw_summary['max_days']:.0f}")
    click.echo(f"% ocean affected: {mhw_summary['pct_affected']:.1f}%")

    dhw = detect.load_or_compute_dhw(sst, region, start, end)
    dhw_summary = analyse.summarise_dhw(dhw, ocean_mask)

    click.echo("\n── DHW Summary ──────────────────")
    click.echo(f"Mean peak DHW: {dhw_summary['mean_peak_dhw']:.1f}")
    click.echo(f"Max DHW: {dhw_summary['max_dhw']:.0f}")
    click.echo(f"% above 4 DHW: {dhw_summary['pct_above_4']:.1f}%")
    click.echo(f"% above 8 DHW: {dhw_summary['pct_above_8']:.1f}%")

    
    click.echo("\n── Monthly SST Anomaly ──────────")
    _, monthly_climatology = detect.load_or_compute_mmm(region)
    anomaly = analyse.monthly_sst_anomaly(sst, monthly_climatology, ocean_mask)
    click.echo(anomaly.to_string())

    # Save analysis data
    # Create output dir if it doesn't exist
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    anomaly.to_csv(os.path.join(output_dir, f"sst_anomaly_{region}_{start}_{end}.csv"))

    with open(os.path.join(output_dir, f"mhw_summary_{region}_{start}_{end}.json"), "w") as f:
        json.dump(mhw_summary, f, indent=2)

    with open(os.path.join(output_dir, f"dhw_summary_{region}_{start}_{end}.json"), "w") as f:
        json.dump(dhw_summary, f, indent=2)

    # save MHW days spatial map as NetCDF
    mhw_days.to_netcdf(os.path.join(output_dir, f"mhw_days_{region}_{start}_{end}.nc"))

    # save DHW spatial map as NetCDF  
    dhw.to_netcdf(os.path.join(output_dir, f"dhw_{region}_{start}_{end}.nc"))

    click.echo(f"Data saved to {output_dir}/")

    click.echo("\n── Generating plots ──────────")

    # plot MHW days
    fig_mhw = visualise.plot_mhw_days(
        mhw_days,
        hotspot_lat=mhw_summary["hotspot_lat"],
        hotspot_lon=mhw_summary["hotspot_lon"],
        title=f"MHW days {start[:4]} — {region}"
    )
    fig_mhw.savefig(os.path.join(output_dir, f"mhw_days_{region}_{start}_{end}.png"), 
                    dpi=256, bbox_inches='tight', pad_inches = 0.1)
    plt.close(fig_mhw)

    # plot DHW
    fig_dhw = visualise.plot_dhw(
        dhw,
        hotspot_lat=dhw_summary["hotspot_lat"],
        hotspot_lon=dhw_summary["hotspot_lon"],
        max_dhw=dhw_summary["max_dhw"],
        title=f"Max DHW {start[:4]} — {region}"
    )
    fig_dhw.savefig(os.path.join(output_dir, f"dhw_{region}_{start}_{end}.png"),
                    dpi=256, bbox_inches='tight', pad_inches = 0.1)
    plt.close(fig_dhw)

    # plot max SST anomaly
    climatology = detect.load_or_compute_climatology(region)
    fig_anom = visualise.plot_sst_anomaly_max(
        sst,
        climatology,
        title=f"Max SST anomaly {start:4} — {region}"
    )
    fig_anom.savefig(
        os.path.join(output_dir, f"sst_anomaly_max_{region}_{start}_{end}.png"),
        dpi=256, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig_anom)

    click.echo(f"\n Plots saved to {output_dir}/")

    # animate DHW progression
    click.echo("\n── Generating DHW animation ──────────")
    #dhw_full = xr.open_dataarray(os.path.join(output_dir, f"dhw_{region}_{start}_{end}.nc"))
    anim = visualise.plot_dhw_animation(dhw, title=f"DHW — {region}")
    anim.save(
        os.path.join(output_dir, f"dhw_animation_{region}_{start}_{end}.mp4"),
        writer="ffmpeg",
        fps=10
    )
    plt.close()
    click.echo("Animation saved!")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    click.echo(f"\n⏱️  Total run time: {minutes}m {seconds}s")

if __name__ == "__main__":
    main()
