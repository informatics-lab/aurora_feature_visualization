import torch
import xarray as xr
import numpy as np
from datetime import datetime

year = 2020
month = 1
day = 1
hour = 0

data_dir = "output.zarr"

surface_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
static_vars = [
    "geopotential",
    "soil_type",
    "land_sea_mask",
]
atmospheric_vars_gcs = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]


time = datetime(year, month, day, hour)
surf_ds = xr.open_zarr(data_dir)
print(surf_ds)
print(surf_ds.time)

surf_ds_time_step = surf_ds.sel(time=time)
surf_data = torch.from_numpy(
    np.array([surf_ds_time_step[var].values for var in surface_vars])
)

print(surf_data.shape)
