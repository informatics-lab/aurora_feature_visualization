import argparse
from pathlib import Path
import os
import torch
import xarray as xr
import numpy as np
from aurora import Batch, Metadata, AuroraSmall, rollout
import matplotlib.pyplot as plt


def main(
    year,
    month,
    day,
    data_dir,
    checkpoint_dir,
    model_name,
    plot_output,
    save_output,
    save_activations,
    output_dir,
    rollout_steps,
):
    # Format the date for folder and file names
    date_str = f"{year}-{month}-{day}"

    # Define the base download path and create a subfolder for the date
    base_download_path = Path(data_dir)
    date_download_path = base_download_path / date_str
    date_download_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    static_vars_ds = xr.open_dataset(date_download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(
        date_download_path / f"{date_str}-surface-level.nc", engine="netcdf4"
    )
    atmos_vars_ds = xr.open_dataset(
        date_download_path / f"{date_str}-atmospheric.nc", engine="netcdf4"
    )

    # Select this time index in the downloaded data.
    i = 1

    # Create a batch
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(
                int(level) for level in atmos_vars_ds.pressure_level.values
            ),
        ),
    )

    # Load model
    model_path = os.path.join(checkpoint_dir, model_name)
    model = AuroraSmall().to("cpu")
    model.load_checkpoint_local(model_path)
    model.eval()

    # Function to hook intermediate activations
    activations = {}

    def hook_fn(module, input, output):
        activations[module.__class__.__name__] = output.clone().detach()

    if save_activations:
        for name, module in model.named_modules():
            module.register_forward_hook(hook_fn)

    # Run inference
    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=rollout_steps)]

    # Save activations if save_activations is True
    if save_activations:
        activations_path = Path(output_dir) / "activations"
        activations_path.mkdir(parents=True, exist_ok=True)
        for name, activation in activations.items():
            activation_file = activations_path / f"{name}.pt"
            torch.save(activation, activation_file)
            print(f"Saved activations for {name} to {activation_file}")

    # Plot predictions if plot_output is True
    if plot_output:
        _, ax = plt.subplots(rollout_steps, 2, figsize=(12, 6.5))

        # print(ax.shape)

        # Make sure ax is shape (rollout_steps, 2)
        if len(ax.shape) == 1:
            ax = np.expand_dims(ax, axis=0)

        for i in range(ax.shape[0]):
            pred = preds[i]

            ax[i, 0].imshow(
                pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50
            )
            ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
            if i == 0:
                ax[i, 0].set_title("Aurora Prediction")
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])

            ax[i, 1].imshow(
                surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50
            )
            if i == 0:
                ax[i, 1].set_title("ERA5")
            ax[i, 1].set_xticks([])
            ax[i, 1].set_yticks([])

        plt.tight_layout()
        plt.show()

    # Save predictions if save_output is True
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for idx, pred in enumerate(preds):
            pred_file = output_path / f"prediction_{idx}.nc"

            lat_values = surf_vars_ds.latitude.values[
                : len(pred.surf_vars["2t"].numpy().squeeze())
            ]  # The model is taking 721 but outputing 720?
            lon_values = surf_vars_ds.longitude.values
            # Correct dimensions
            xr.Dataset(
                {
                    "2t": (
                        ("lat", "lon"),
                        pred.surf_vars["2t"].numpy().squeeze() - 273.15,
                    ),
                },
                coords={
                    "lat": lat_values,
                    "lon": lon_values,
                },
            ).to_netcdf(pred_file)

            print(f"Saved prediction to {pred_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora inference script")

    default_args = {
        "year": "2023",
        "month": "01",
        "day": "01",
        "data_dir": "./data/input_data/era5",
        "checkpoint_dir": "./checkpoints",
        "output_dir": "./output_data",
        "model_name": "aurora-0.25-small-pretrained.ckpt",
        "plot_output": False,
        "save_output": False,
        "save_activations": False,
        "rollout_steps": 1,
    }

    parser.add_argument(
        "--year",
        type=str,
        default=default_args["year"],
        help=f"Year for the data (default: {default_args['year']})",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=default_args["month"],
        help=f"Month for the data (default: {default_args['month']})",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=default_args["day"],
        help=f"Day for the data (default: {default_args['day']})",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_args["data_dir"],
        help=f"Directory where data is stored (default: {default_args['data_dir']})",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=default_args["checkpoint_dir"],
        help=f"Directory where model checkpoints are stored (default: {default_args['checkpoint_dir']})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_args["output_dir"],
        help=f"Directory to save output files (default: {default_args['output_dir']})",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=default_args["model_name"],
        help=f"Model checkpoint file name (default: {default_args['model_name']})",
    )
    parser.add_argument(
        "--plot_output",
        type=bool,
        default=default_args["plot_output"],
        help=f"Whether to plot the model output (default: {default_args['plot_output']})",
    )
    parser.add_argument(
        "--save_output",
        type=bool,
        default=default_args["save_output"],
        help=f"Whether to save the model output to files (default: {default_args['save_output']})",
    )
    parser.add_argument(
        "--save_activations",
        type=bool,
        default=default_args["save_activations"],
        help=f"Whether to save the intermediate activations (default: {default_args['save_activations']})",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=default_args["rollout_steps"],
        help=f"Number of steps to run in the rollout (default: {default_args['rollout_steps']})",
    )

    args = parser.parse_args()

    main(
        args.year,
        args.month,
        args.day,
        args.data_dir,
        args.checkpoint_dir,
        args.model_name,
        args.plot_output,
        args.save_output,
        args.save_activations,
        args.output_dir,
        args.rollout_steps,
    )
