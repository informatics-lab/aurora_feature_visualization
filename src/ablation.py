from aurora import AuroraSmall, Batch, Metadata
import torch
import xarray as xr
from pathlib import Path

import os
import cdsapi
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

year = "2018"
month = "06"
day = "01"
date_str = f"{year}-{month}-{day}"
time_points = ["00:00", "06:00", "12:00", "18:00"]

data_dir = "~/data/input_data/era5"
data_dir = "data/input_data/era5"
base_download_path = Path(data_dir).expanduser()
date_download_path = base_download_path / date_str
date_download_path.mkdir(parents=True, exist_ok=True)


def download_if_not_exists(c, file_path, request_type, request_params):
    """
    Helper function to download data if the file doesn't already exist.
    """
    if not file_path.exists():
        c.retrieve(request_type, request_params, str(file_path))
        print(f"Downloaded: {file_path.name}")
    else:
        print(f"File already exists: {file_path.name}")

    mlflow.log_artifact(file_path)


url = "url: https://cds.climate.copernicus.eu/api"
key = "key: 96ef2978-f538-44d8-99e3-0f4e3db08554"

with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
    f.write("\n".join([url, key]))

with open(os.path.expanduser("~/.cdsapirc")) as f:
    print(f.read())

c = cdsapi.Client()

# Ensure zero-padding for month and day
date_str = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
date_download_path = base_download_path / date_str
date_download_path.mkdir(parents=True, exist_ok=True)

print(f"Processing data for {date_str}...")

# Download static variables
static_file = date_download_path / f"{date_str}_static.nc"
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
surface_file = date_download_path / f"{date_str}_surface.nc"
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
atmospheric_file = date_download_path / f"{date_str}_atmospheric.nc"
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

# Load datasets
static_vars_ds = xr.open_dataset(
    date_download_path / f"{date_str}_static.nc", engine="netcdf4"
)
surf_vars_ds = xr.open_dataset(
    date_download_path / f"{date_str}_surface.nc", engine="netcdf4"
)
atmos_vars_ds = xr.open_dataset(
    date_download_path / f"{date_str}_atmospheric.nc", engine="netcdf4"
)

# Select this time index in the downloaded data.
i = 1

# Create a batch
batch = Batch(
    surf_vars={
        "2t": torch.from_numpy(
            surf_vars_ds["t2m"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "10u": torch.from_numpy(
            surf_vars_ds["u10"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "10v": torch.from_numpy(
            surf_vars_ds["v10"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "msl": torch.from_numpy(
            surf_vars_ds["msl"].values[[i - 1, i]][None]
        ).requires_grad_(),
    },
    static_vars={
        "z": torch.from_numpy(static_vars_ds["z"].values[0]).requires_grad_(),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]).requires_grad_(),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]).requires_grad_(),
    },
    atmos_vars={
        "t": torch.from_numpy(
            atmos_vars_ds["t"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "u": torch.from_numpy(
            atmos_vars_ds["u"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "v": torch.from_numpy(
            atmos_vars_ds["v"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "q": torch.from_numpy(
            atmos_vars_ds["q"].values[[i - 1, i]][None]
        ).requires_grad_(),
        "z": torch.from_numpy(
            atmos_vars_ds["z"].values[[i - 1, i]][None]
        ).requires_grad_(),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

model = AuroraSmall(use_lora=False, autocast=True)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model.eval()
model.configure_activation_checkpointing()

prediction = model.forward(batch)
torch.save(prediction, "prediction.pt")
mlflow.log_artifact("prediction.pt")

neuron_idx = 7
with torch.no_grad():
    model.backbone.encoder_layers[0]._checkpoint_wrapped_module.blocks[
        0
    ].mlp.fc2.weight[neuron_idx].zero_()
    model.backbone.encoder_layers[0]._checkpoint_wrapped_module.blocks[0].mlp.fc2.bias[
        neuron_idx
    ].zero_()

prediction_abalated = model.forward(batch)
torch.save(prediction_abalated, "prediction_abalated.pt")
mlflow.log_artifact("prediction_abalated.pt")
