import click
from reefwatch import fetch, detect, analyse
from reefwatch.constants import ANALYSIS_START, ANALYSIS_END

@click.command()
# Region for analysis, default is coral_triangle
@click.option("--region", default="coral_triangle", show_default=True, help="Region to analyse")
@click.option("--start", default=ANALYSIS_START, show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=ANALYSIS_END, show_default=True, help="End date (YYYY-MM-DD)")
@click.option("--comparison_years", default="1991,2023", show_default=True, 
              help="Comma-separated years for MHW comparison (e.g. 1991,2023)")
def main(region, start, end, comparison_years):
    """🪸 Reefwatch — coral bleaching thermal stress analysis."""
    click.echo(f"\n🌊 Fetching SST data for {region} from {start} to {end}...")
    
    ds = fetch.fetch_data(region,start,end)

    sst = ds["analysed_sst"]
    ocean_mask = ~sst.isel(time=0).isnull().compute()  # False where NaN (land), True where has data (ocean)

    mhw = detect.compute_mhw(sst, region)
    mhw_days = mhw["mhw"].sum(dim="time")
    mhw_summary = analyse.summarise_mhw(mhw_days, ocean_mask)
    click.echo("\n── MHW Summary ──────────────────")
    click.echo(f"Mean MHW days: {mhw_summary['mean_days']:.1f}")
    click.echo(f"Median MHW days: {mhw_summary['median_days']:.1f}")
    click.echo(f"Max MHW days: {mhw_summary['max_days']:.0f}")
    click.echo(f"% ocean affected: {mhw_summary['pct_affected']:.1f}%")

    dhw = detect.compute_dhw(sst, region)
    dhw_summary = analyse.summarise_dhw(dhw, ocean_mask)

    click.echo("\n── DHW Summary ──────────────────")
    click.echo(f"Mean peak DHW: {dhw_summary['mean_peak_dhw']:.1f}")
    click.echo(f"Max DHW: {dhw_summary['max_dhw']:.0f}")
    click.echo(f"% above 4 DHW: {dhw_summary['pct_above_4']:.1f}%")
    click.echo(f"% above 8 DHW: {dhw_summary['pct_above_8']:.1f}%")


    click.echo("\n── Year Comparison ──────────────")
    years = [int(y) for y in comparison_years.split(",")]
    year_comparison = analyse.compare_years(region, years, ocean_mask)
    click.echo(year_comparison.to_string())

    
    click.echo("\n── Monthly SST Anomaly ──────────")
    anomaly = analyse.monthly_sst_anomaly(sst, region, ocean_mask)
    click.echo(anomaly.to_string())

if __name__ == "__main__":
    main()
