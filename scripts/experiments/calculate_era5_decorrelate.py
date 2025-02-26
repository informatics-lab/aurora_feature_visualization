import torch
from pathlib import Path
import xarray as xr
from tqdm.contrib.itertools import product
import numpy as np

years = ["2018", "2019"]
months = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
days = ["1", "15"]

base_download_path = Path("data/input_data/era5")
base_download_path.mkdir(parents=True, exist_ok=True)

surface_vars = [
    "t2m",
    "u10",
    "v10",
    "msl",
]
static_vars = [
    "z",
    "lsm",
    "slt",
]
atmospheric_vars = [
    "t",
    "u",
    "v",
    "z",
    "q",
]

cov_mtx_surface = 0.0
cov_mtx_static = 0.0
cov_mtx_atmospheric = 0.0
num_days = len(years) * len(months) * len(days)


def rgb_cov(im, num_vars):
    im_re = im.reshape(-1, num_vars)
    im_re -= im_re.mean(0, keepdim=True)
    return 1 / (im_re.shape[0] - 1) * im_re.T @ im_re


for year, month, day in product(years, months, days):
    date_str = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    surface_ds = xr.open_dataset(base_download_path / f"{date_str}_surface.nc")
    static_ds = xr.open_dataset(base_download_path / f"{date_str}_static.nc")
    atmospheric_ds = xr.open_dataset(base_download_path / f"{date_str}_atmospheric.nc")

    for time_step in surface_ds.valid_time:
        surface_ds_time_step = surface_ds.sel(valid_time=time_step)
        surface_data = torch.from_numpy(
            np.array([surface_ds_time_step[var].values for var in surface_vars])
        )
        cov_mtx_surface += rgb_cov(surface_data.permute(1, 2, 0), len(surface_vars))

    static_data = torch.from_numpy(
        np.array([static_ds[var].values for var in static_vars])
    )
    cov_mtx_static += rgb_cov(
        static_data.permute(1, 2, 3, 0).squeeze(0), len(static_vars)
    )

    for time_step in atmospheric_ds.valid_time:
        atmospheric_ds_time_step = atmospheric_ds.sel(valid_time=time_step)
        atmospheric_data = torch.from_numpy(
            np.array([atmospheric_ds_time_step[var].values for var in atmospheric_vars])
        )
        cov_mtx_atmospheric += rgb_cov(
            atmospheric_data.permute(1, 2, 3, 0), len(atmospheric_vars)
        )

num_surface_time_steps = num_days * len(surface_ds.valid_time)
cov_mtx_surface = cov_mtx_surface / num_surface_time_steps

cov_mtx_static = cov_mtx_static / num_days

num_atmospheric_time_steps = num_days * len(atmospheric_ds.valid_time)
cov_mtx_atmospheric = cov_mtx_atmospheric / num_atmospheric_time_steps

print("Surface Covariance Matrix:")
print(cov_mtx_surface)
print("Static Covariance Matrix:")
print(cov_mtx_static)
print("Atmospheric Covariance Matrix:")
print(cov_mtx_atmospheric)
