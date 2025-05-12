import xarray as xr
import numpy as np
import argparse
import pandas as pd
from time import time
from itertools import product
import zarr
from zarr.convenience import consolidate_metadata
import os

static_vars = ["geopotential_at_surface", "land_sea_mask", "soil_type"]
surface_vars = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]
upper_vars = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]


def process_and_save_data(
    output_path,
    start_date=None,
    end_date=None,
    years=None,
    months=None,
    days=None,
    times=None,
    mode="w",
):
    total_start_time = time()

    print("Loading data...")
    load_start_time = time()
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        chunks={"time": "auto"},
        decode_times=True,
        decode_cf=True,
        decode_coords=True,
        consolidated=True,
    )
    load_end_time = time()
    print(
        f"Data loading complete - Time Taken: {load_end_time - load_start_time:.2f} seconds"
    )

    print("Selecting data...")
    select_start_time = time()
    if start_date and end_date:
        ds_filtered = ds.sel(time=slice(start_date, end_date))

    elif years and months and days and times:
        datetime_strings = []
        for year, month, day, time_str in product(years, months, days, times):
            datetime_strings.append(f"{year:04d}-{month:02d}-{day:02d} {time_str}")

        try:
            datetime_index = pd.to_datetime(datetime_strings)
            ds_filtered = ds.sel(
                time=datetime_index,
            )
        except ValueError as e:
            print(f"Error creating datetime index: {e}")
            return
    select_end_time = time()
    print(
        f"Data selection complete - Time Taken: {select_end_time - select_start_time:.2f} seconds"
    )

    print("Saving data...")
    save_start_time = time()
    selected_data = ds_filtered[surface_vars + upper_vars + static_vars].astype(
        np.float32
    )

    if mode == "a" and os.path.exists(output_path):
        existing_ds = xr.open_zarr(output_path, consolidated=True)
        new_times = selected_data.time.values
        existing_times = existing_ds.time.values
        unique_new_times = np.setdiff1d(new_times, existing_times)
        if len(unique_new_times) > 0:
            selected_data = selected_data.sel(time=unique_new_times)
            if len(selected_data.time) > 0:
                selected_data.to_zarr(output_path, mode="a", append_dim="time")
                consolidate_metadata(output_path)
            else:
                print("All times already exist. No new data was added.")
        else:
            print("All times already exist. No new data was added.")

    else:
        selected_data.to_zarr(
            output_path,
            mode=mode,
            # encoding=encoding,
        )
        consolidate_metadata(output_path)
    save_end_time = time()
    print(
        f"Data saving complete - Time Taken: {save_end_time - save_start_time:.2f} seconds"
    )

    total_end_time = time()
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process weather data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format")
    group.add_argument(
        "--years", nargs="+", type=int, help="List of years (e.g., 2020 2021)"
    )
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--months", nargs="+", type=int, help="List of months (e.g., 1 2 3)"
    )
    parser.add_argument(
        "--days", nargs="+", type=int, help="List of days (e.g., 1 15 30)"
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=str,
        default=["00:00", "06:00", "12:00", "18:00"],
        help="List of times (e.g., 00:00 06:00 12:00)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/input_data/era5.zarr",
        help="Output Zarr file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["w", "a"],
        default="w",
        help="Write mode: 'w' (write/overwrite) or 'a' (append)",
    )

    args = parser.parse_args()

    if args.start_date:
        if not args.end_date:
            print("Error: end_date is required with start_date")
        else:
            process_and_save_data(
                args.output_path,
                start_date=args.start_date,
                end_date=args.end_date,
                mode=args.mode,
            )
    else:
        if not (args.months and args.days and args.times):
            print("Error: months, days, and times are required with years")
        else:
            process_and_save_data(
                args.output_path,
                years=args.years,
                months=args.months,
                days=args.days,
                times=args.times,
                mode=args.mode,
            )
