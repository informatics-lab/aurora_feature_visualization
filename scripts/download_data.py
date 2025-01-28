from pathlib import Path
from datetime import datetime
import cdsapi

# Initialize the CDS API client
c = cdsapi.Client()

# Set date and time variables
year = "2023"
month = "01"
day = "01"
time_points = ["00:00", "06:00", "12:00", "18:00"]

# Format the date for folder and file names
date_str = f"{year}-{month}-{day}"

# Define the base download path and create a subfolder for the date
base_download_path = Path("./input_data")
date_download_path = base_download_path / date_str
date_download_path.mkdir(parents=True, exist_ok=True)

def download_if_not_exists(file_path, request_type, request_params):
    """
    Helper function to download data if the file doesn't already exist.
    """
    if not file_path.exists():
        c.retrieve(request_type, request_params, str(file_path))
        print(f"Downloaded: {file_path.name}")
    else:
        print(f"File already exists: {file_path.name}")

# Download the static variables
static_file = date_download_path / "static.nc"
download_if_not_exists(
    static_file,
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "geopotential",
            "land_sea_mask",
            "soil_type",
        ],
        "year": year,
        "month": month,
        "day": day,
        "time": "00:00",
        "format": "netcdf",
    },
)

# Download the surface-level variables
surface_file = date_download_path / f"{date_str}-surface-level.nc"
download_if_not_exists(
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

# Download the atmospheric variables
atmospheric_file = date_download_path / f"{date_str}-atmospheric.nc"
download_if_not_exists(
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
