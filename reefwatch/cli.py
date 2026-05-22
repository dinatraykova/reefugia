import click
from reefwatch import fetch, analyse

@click.command()
# Region for analysis, default is coral_triangle
@click.option("--region", default="coral_triangle", show_default=True, help="Region to analyse")
# Time period 2023 coral bleaching event — worst on record globally, severely impacted the Coral Triangle
@click.option("--start", default="2023-01-01", show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default="2023-09-01", show_default=True, help="End date (YYYY-MM-DD)")
def main(region, start, end):
    """🪸 Reefwatch — coral bleaching thermal stress analysis."""
    click.echo(f"\n🌊 Fetching SST data for {region} from {start} to {end}...")
    
    ds = fetch.fetch_data(region,start,end)
    click.echo(ds)
    #data = fetch.??
    #analyse.??

if __name__ == "__main__":
    main()