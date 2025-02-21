from pathlib import Path
import cdsapi
import argparse


def download_if_not_exists(c, file_path, request_type, request_params):
    """
    Helper function to download data if the file doesn't already exist.
    """
    if not file_path.exists():
        c.retrieve(request_type, request_params, str(file_path))
        print(f"Downloaded: {file_path.name}")
    else:
        print(f"File already exists: {file_path.name}")


def download_era5_data(year, month, day, time_points, base_download_path):
    """Downloads ERA5 data for the specified date and time points."""

    date_str = f"{year}-{month}-{day}"
    date_download_path = base_download_path / date_str
    date_download_path.mkdir(parents=True, exist_ok=True)

    c = (
        cdsapi.Client()
    )  # Initialize client inside the function for better encapsulation

    # Download static variables
    static_file = date_download_path / "static.nc"
    download_if_not_exists(
        c,
        static_file,
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask", "soil_type"],
            "year": year,
            "month": month,
            "day": day,
            "time": "00:00",
            "format": "netcdf",
        },
    )

    # Download surface-level variables
    surface_file = date_download_path / f"{date_str}-surface-level.nc"
    download_if_not_exists(
        c,
        surface_file,
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": time_points,
            "format": "netcdf",
        },
    )

    # Download atmospheric variables
    atmospheric_file = date_download_path / f"{date_str}-atmospheric.nc"
    download_if_not_exists(
        c,
        atmospheric_file,
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                "geopotential",
            ],
            "pressure_level": [
                "50",
                "100",
                "150",
                "200",
                "250",
                "300",
                "400",
                "500",
                "600",
                "700",
                "850",
                "925",
                "1000",
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": time_points,
            "format": "netcdf",
        },
    )

    print("All requested data has been processed!")


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 data.")
    parser.add_argument("--years", help="Year of data to download (YYYY)")
    parser.add_argument(
        "--months", type=str, nargs="+", help="Month of data to download (MM)"
    )
    parser.add_argument(
        "--days", type=str, nargs="+", help="Day of data to download (DD)"
    )
    parser.add_argument(
        "--times",
        type=str,
        nargs="+",
        default=["00:00", "06:00", "12:00", "18:00"],
        help="Time points to download (HH:MM), space-separated",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/input_data/era5",
        help="Base directory to save downloaded data",
    )
    args = parser.parse_args()

    base_download_path = Path(args.output_dir)
    download_era5_data(
        args.years, args.months, args.days, args.times, base_download_path
    )


if __name__ == "__main__":
    main()
