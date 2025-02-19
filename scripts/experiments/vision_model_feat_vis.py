import torch
import numpy as np
import torchvision.models as models
from tqdm import trange
import matplotlib.pyplot as plt
from hooks import hook_specific_layer
import image
import transform
import logging
import argparse
import os
from typing import Callable, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def neuron_loss(tensor: torch.Tensor, neuron_idx: int) -> torch.Tensor:
    """
    Compute the negative mean activation for a given neuron.
    """
    return -tensor[:, :, :, neuron_idx].mean()


def attention_loss(tensor: torch.Tensor, head_idx: int) -> torch.Tensor:
    """
    Compute the negative mean attention for a given attention head.
    """
    return -tensor[:, head_idx].mean()


def build_plugin_transforms(focus_value: int, image_size: int) -> Callable:
    """
    Build the transforms for plugin inversion.
    The focus value is updated every iteration.
    """
    transforms_list: List[Callable] = [
        transform.focus(focus_value, 0),
        transform.jitter(8),
        transform.color_jitter_r(1, True),
    ]
    return transform.compose(transforms_list)


def plugin_in_inversion(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    image_f: Callable[[], torch.Tensor],
    image_size: int,
    n_epochs: int,
    hook,
    neuron_idx: int,
) -> None:
    """
    Plugin inversion stage for feature visualisation.
    """
    model.eval()
    step = image_size // 8
    for i, size in enumerate(range(2 * step, image_size + 1, step)):
        cumulative_loss = 0.0
        pbar = trange(n_epochs, desc=f"PII Step {i + 1} Loss: -")
        for epoch in pbar:
            optimizer.zero_grad()

            # Create a new transform with current focus value.
            cur_transform = build_plugin_transforms(size, image_size)
            _ = model(cur_transform(image_f()))

            loss = neuron_loss(hook.features, neuron_idx)
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"PII Step {i + 1} Loss: {cumulative_loss / (epoch + 1):.3f}"
            )


def fourier_fv(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    image_f: Callable[[], torch.Tensor],
    image_size: int,
    n_epochs: int,
    hook,
    neuron_idx: int,
) -> None:
    """
    Fourier feature visualization training loop.
    """
    transforms_list: List[Callable] = [
        transform.jitter(8),
        transform.random_scale_vit(
            [1 + (i - 5) / 50.0 for i in range(11)],
            target_size=(image_size, image_size),
        ),
        transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
        transform.jitter(4),
    ]
    transform_f = transform.compose(transforms_list)

    cumulative_loss = 0.0
    pbar = trange(n_epochs, desc="FourierFV Loss: -")
    for epoch in pbar:
        optimizer.zero_grad()

        _ = model(transform_f(image_f()))

        loss = neuron_loss(hook.features, neuron_idx)
        cumulative_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"FourierFV Loss: {cumulative_loss / (epoch + 1):.3f}")


def plot_images(
    images: List[np.ndarray], titles: List[str], layer_name: str, neuron_idx: int
) -> None:
    """
    Plot the feature visualization results side by side.
    """
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(
        f"Feature Visualisation: {layer_name} - Neuron {neuron_idx}", fontsize=16
    )
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_images(
    images: List[np.ndarray],
    titles: List[str],
    output_dir: str,
    layer_name: str,
    neuron_idx: int,
) -> None:
    """
    Save the feature visualization images to disk.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for img, title in zip(images, titles):
        # File name format includes layer, neuron index, and title.
        filename = f"{layer_name.replace('.', '_')}_neuron{neuron_idx}_{title}.png"
        filepath = os.path.join(output_dir, filename)
        plt.imsave(filepath, img)
        logging.info(f"Saved image: {filepath}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Feature visualisation with activation maximisation"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--fourier_epochs", type=int, default=500)
    parser.add_argument("--pii_epochs", type=int, default=150)
    parser.add_argument("--pii_fourier_epochs", type=int, default=200)
    parser.add_argument("--layer_name", type=str, default="features.1.0")
    parser.add_argument("--neuron_idx", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--no_plot", action="store_false", help="Plot the images")
    parser.add_argument("--save", action="store_true", help="Save the images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output_data/fv_images",
        help="Output directory for saving",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1).to(device)
    layer_name = args.layer_name

    hook = hook_specific_layer(model, layer_name)
    image_size = args.image_size
    # Initialize the image with FFT parameters.
    params, image_f = image.image(image_size, fft=True, decorrelate=True, device=device)
    neuron_idx = args.neuron_idx

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # First stage: Fourier Feature Visualisation Optimisation.
    logging.info("Starting Fourier Feature Visualisation optimisation...")
    fourier_fv(
        model, optimizer, image_f, image_size, args.fourier_epochs, hook, neuron_idx
    )
    first_stage_image = image_f().clone().detach().cpu().numpy()
    first_stage_image = np.transpose(np.squeeze(first_stage_image), (1, 2, 0))

    # Second stage: Plugin Inversion Inversion Optimisation.
    # Reinitialize the image parameters.
    params, image_f = image.image(image_size, fft=True, decorrelate=True, device=device)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    logging.info("Starting Plugin Inversion optimisation...")
    plugin_in_inversion(
        model, optimizer, image_f, image_size, args.pii_epochs, hook, neuron_idx
    )
    second_stage_image = image_f().clone().detach().cpu().numpy()
    second_stage_image = np.transpose(np.squeeze(second_stage_image), (1, 2, 0))

    # Third stage: Continue with Fourier feature visualization.
    logging.info("Starting Fourier Feature Visualisation (post PII) optimisation...")

    fourier_fv(
        model,
        optimizer,
        image_f,
        image_size,
        args.pii_fourier_epochs,
        hook,
        neuron_idx,
    )
    third_stage_image = image_f().clone().detach().cpu().numpy()
    third_stage_image = np.transpose(np.squeeze(third_stage_image), (1, 2, 0))

    hook.close()

    if args.save:
        save_images(
            images=[first_stage_image, second_stage_image, third_stage_image],
            titles=["fourierfv", "pii", "pii_fourierfv"],
            output_dir=args.output_dir,
            layer_name=layer_name,
            neuron_idx=neuron_idx,
        )

    # Plot the images.
    if args.no_plot:
        plot_images(
            images=[first_stage_image, second_stage_image, third_stage_image],
            titles=["FourierFV", "PII", "PII + FourierFV"],
            layer_name=layer_name,
            neuron_idx=neuron_idx,
        )


if __name__ == "__main__":
    main()
