import os
import torch
import argparse
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
import matplotlib.pyplot as plt
from tqdm import trange
from hooks import hook_specific_layer
from transform import jitter_3d
import image


def build_era_image(lat, lon, time, lvl_type, device):
    surf_params, surf_image_f = image.image(
        lat, lon, time, lvl_type="surf", device=device
    )
    static_params, static_image_f = image.image(
        lat, lon, time, lvl_type="static", device=device
    )
    atmos_params, atmos_image_f = image.image(
        lat, lon, time, lvl_type="atmos", device=device
    )
    params = surf_params + static_params + atmos_params
    return params, [surf_image_f, static_image_f, atmos_image_f]


def build_batch(image_fs, lat, lon, device, transform):
    surf_vars = {
        k: transform(image_fs[0]()[:, :, i]).squeeze(axis=2)
        for i, k in enumerate(("2t", "10u", "10v", "msl"))
    }
    static_vars = {
        k: image_fs[1]()[:, 0, i].squeeze(axis=[0, 1, 2])
        for i, k in enumerate(("lsm", "z", "slt"))
    }
    atmos_vars = {
        k: image_fs[2]()[:, :, i].squeeze(axis=2)
        for i, k in enumerate(("z", "u", "v", "t", "q"))
    }

    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=torch.linspace(90, -90, lat, device=device),
            lon=torch.linspace(0, 360, lon + 1, device=device)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(
                700,
                850,
                925,
                1000,
            ),
        ),
    )
    return batch


def neuron_loss(tensor: torch.Tensor, neuron_idx: int) -> torch.Tensor:
    return -tensor[:, :, neuron_idx].mean()


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AuroraSmall(
        use_lora=False,
        autocast=True,
    )
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    model.configure_activation_checkpointing()
    model.to(device)
    model.eval()

    hook = hook_specific_layer(model, args.layer_name)

    transform = jitter_3d(5)
    loss_fn = neuron_loss

    # Build ERA image
    params, image_fs = build_era_image(
        args.lat, args.lon, args.time, args.lvl_type, device
    )
    optimizer = torch.optim.Adam(
        params, lr=args.learning_rate, betas=(0.5, 0.99), eps=1e-8
    )

    pbar = trange(args.n_epochs, desc="loss: -")
    for _ in pbar:
        optimizer.zero_grad()
        batch = build_batch(image_fs, args.lat, args.lon, device, transform)
        _ = model(batch)

        # Compute loss using the neuron index from arguments.
        loss = loss_fn(hook.features, args.neuron_idx)

        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss:.2f}")

    batch = build_batch(image_fs, args.lat, args.lon, device, transform)
    rollout_steps = 2
    surface_names = ["2t", "10u", "10v", "msl"]
    surface_vars = [
        batch.surf_vars["2t"],
        batch.surf_vars["10u"],
        batch.surf_vars["10v"],
        batch.surf_vars["msl"],
    ]
    fig, axes = plt.subplots(
        len(surface_vars),
        rollout_steps,
        figsize=(15, len(surface_vars) * 4),
        constrained_layout=True,
        squeeze=False,
    )
    for i, (name, var_data) in enumerate(zip(surface_names, surface_vars)):
        for j in range(rollout_steps):
            ax = axes[i, j]

            img = var_data[0, j].detach().cpu().numpy()
            _ = ax.imshow(img, cmap="coolwarm", origin="lower")
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")

            title_time = "t - 1" if j == 0 else "t"
            ax.set_title(f"{name} ({title_time})", pad=10)

    fig.suptitle(f"{args.layer_name} - neuron idx: {args.neuron_idx}", fontsize=16)

    if args.save_output:
        # Replace periods in the layer name to create a safe folder name.
        safe_layer_name = args.layer_name.replace(".", "_")
        output_dir = os.path.join(
            "data", "output_data", "feature_visualizations", f"{safe_layer_name}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"neuron_{args.neuron_idx}.png")
        fig.savefig(output_file)
        print(f"Figure saved to {output_file}")

    if args.plot_output:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aurora Model Neuron Optimization", add_help=True
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="backbone.encoder_layers.0._checkpoint_wrapped_module.blocks.0.mlp",
        help="Name of the layer to optimize",
    )
    parser.add_argument(
        "--neuron_idx", type=int, default=1, help="Index of the neuron to optimize"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed to initialise the PRNG"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-2,
        help="Learning rate for the optimizer",
    )
    parser.add_argument("--lat", type=int, default=180, help="Latitude resolution")
    parser.add_argument("--lon", type=int, default=360, help="Longitude resolution")
    parser.add_argument("--time", type=int, default=2, help="Time steps")
    parser.add_argument(
        "--lvl_type",
        type=str,
        default="surf",
        choices=["surf", "atmos", "static"],
        help="Level type to use (surf, atmos, or static)",
    )
    parser.add_argument(
        "--no-plot_output",
        dest="plot_output",
        action="store_false",
        help=f"Whether to plot the model output",
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help=f"Whether to save the model output to files)",
    )
    args = parser.parse_args()
    main(args)
