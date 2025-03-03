import torch
import xarray as xr
from tqdm.contrib.itertools import product
import numpy as np
from datetime import datetime

years = [2019]
months = [
    1,
    2,
    3,
    4,
    5,
    6,
]
days = [1, 15]

# years = [2018]
# months = [1]
# days = [1]

data_dir = "data/input_data/era5.zarr"

cov_mtx_surface = 0.0
cov_mtx_static = 0.0
cov_mtx_atmospheric = 0.0
num_days = len(years) * len(months) * len(days)


def rgb_cov(im, num_vars):
    im_re = im.reshape(-1, num_vars)
    im_re -= im_re.mean(0, keepdim=True)
    return 1 / (im_re.shape[0] - 1) * im_re.T @ im_re


surf_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
static_vars = [
    "geopotential_at_surface",
    "soil_type",
    "land_sea_mask",
]
atmos_vars = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

ds = xr.open_zarr(data_dir)
for year, month, day in product(years, months, days):
    date_str = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    time_start = datetime(year, month, day, hour=0)
    time_end = datetime(year, month, day, hour=18)
    ds_time_step = ds.sel(time=slice(time_start, time_end))

    for time_step in ds_time_step.time:
        surf_ds_time_step = ds_time_step.sel(time=time_step)
        surf_data = torch.from_numpy(
            np.array([surf_ds_time_step[var].values for var in surf_vars])
        )
        cov_mtx_surface += rgb_cov(surf_data.permute(1, 2, 0), len(surf_vars))

    static_data = torch.from_numpy(
        np.array([ds_time_step[var].values for var in static_vars])
    )
    cov_mtx_static += rgb_cov(static_data.permute(1, 2, 0).squeeze(0), len(static_vars))

    for time_step in ds_time_step.time:
        atmos_ds_time_step = ds_time_step.sel(time=time_step)
        atmos_data = torch.from_numpy(
            np.array([atmos_ds_time_step[var].values for var in atmos_vars])
        )
        cov_mtx_atmospheric += rgb_cov(atmos_data.permute(1, 2, 3, 0), len(atmos_vars))

num_time_steps = num_days * len(ds_time_step.time)
cov_mtx_surface = cov_mtx_surface / num_time_steps

cov_mtx_static = cov_mtx_static / num_days

cov_mtx_atmospheric = cov_mtx_atmospheric / num_time_steps

print("Surface Covariance Matrix:")
print(cov_mtx_surface)
print("Static Covariance Matrix:")
print(cov_mtx_static)
print("Atmospheric Covariance Matrix:")
print(cov_mtx_atmospheric)
