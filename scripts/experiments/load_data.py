import torch
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime

surface_vars = [
    "t2m",
    "u10",
    "v10",
    "msl",
]
static_vars = [
    "z",
    "slt",
    "lsm",
]
atmospheric_vars = [
    "t",
    "u",
    "v",
    "q",
    "z",
]

year = 2018
month = 1
day = 1
data_dir = "data/input_data/era5/"

# Format the date for folder and file names
date_str = f"{year}-{month:02}-{day:02}"

# Define the base download path and create a subfolder for the date
base_download_path = Path(data_dir)
base_download_path.mkdir(parents=True, exist_ok=True)

# Load datasets
static_ds = xr.open_dataset(
    base_download_path / f"{date_str}_static.nc", engine="netcdf4"
)
surf_ds = xr.open_dataset(
    base_download_path / f"{date_str}_surface.nc", engine="netcdf4"
)
atmos_ds = xr.open_dataset(
    base_download_path / f"{date_str}_atmospheric.nc", engine="netcdf4"
)

print(surf_ds)


time = datetime(year, month, day, 0, 0)
surf_ds_time_step = surf_ds.sel(valid_time=time)
surf_data = torch.from_numpy(
    np.array([surf_ds_time_step[var].values for var in surface_vars])
)

print(surf_data.shape)
