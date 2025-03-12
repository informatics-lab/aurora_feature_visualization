import torch
import numpy as np
import torchvision.models as models
from torchvision.transforms import transforms
from tqdm import trange
import matplotlib.pyplot as plt
import argparse
import os
import logging
from typing import Callable
import image

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def neuron_loss(tensor: torch.Tensor, neuron_idx: int) -> torch.Tensor:
    """Compute the negative mean activation for a given neuron."""
    return -tensor[:, :, :, neuron_idx].mean()


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
    Applies simple jitter and random rotation transforms.
    """
    # For simplicity, we use a fixed set of transforms.
    transform_f = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    cumulative_loss = 0.0
    pbar = trange(n_epochs, desc="FourierFV Loss: -")
    for epoch in pbar:
        optimizer.zero_grad()

        # Apply transformation and run the model.
        img = image_f()
        transformed_img = transform_f(img)
        _ = model(transformed_img)

        loss = neuron_loss(hook.features, neuron_idx)
        cumulative_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"FourierFV Loss: {cumulative_loss / (epoch + 1):.4f}")


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize each channel of the image to the [0, 1] range using min-max scaling.
    """
    norm_img = np.empty_like(img)
    # Assume image shape is (H, W, C)
    for i in range(img.shape[-1]):
        channel = img[..., i]
        # Avoid division by zero if the channel has constant value
        if channel.max() - channel.min() != 0:
            norm_img[..., i] = (channel - channel.min()) / (
                channel.max() - channel.min()
            )
        else:
            norm_img[..., i] = channel
    return norm_img


# def plot_image(img: np.ndarray, title: str, layer_name: str, neuron_idx: int) -> None:
#     """Plot the feature visualization result."""
#     img = normalize_image(img)
#     plt.figure(figsize=(5, 5))
#     plt.imshow(img)
#     plt.title(f"{layer_name} - Neuron {neuron_idx} ({title})")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


def plot_image(img: np.ndarray, title: str, layer_name: str, neuron_idx: int) -> None:
    """
    Plot the feature visualization result for each of the three channels as separate grayscale subplots.
    """
    # Normalize the image first
    img = normalize_image(img)

    if img.shape[-1] != 3:
        raise ValueError("Expected image with three channels.")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{layer_name} - Neuron {neuron_idx} ({title})", fontsize=16)

    channel_names = ["Channel 1", "Channel 2", "Channel 3"]
    for i, ax in enumerate(axs):
        # Extract channel and display as grayscale image.
        channel_img = img[..., i]
        ax.imshow(channel_img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(channel_names[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def save_image(
    img: np.ndarray, title: str, output_dir: str, layer_name: str, neuron_idx: int
) -> None:
    """Save the feature visualization image to disk."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    filename = f"{layer_name.replace('.', '_')}_neuron{neuron_idx}_{title}.png"
    filepath = os.path.join(output_dir, filename)
    plt.imsave(filepath, img)
    logging.info(f"Saved image: {filepath}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simplified Fourier Feature Visualisation"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--image_size", type=int, default=100, help="Input image size")
    parser.add_argument("--fourier_epochs", type=int, default=200)
    parser.add_argument("--layer_name", type=str, default="features.1.0")
    parser.add_argument("--neuron_idx", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--no_plot", action="store_false", help="Do not plot the image")
    parser.add_argument("--save", action="store_true", help="Save the image")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_images",
        help="Output directory for saving",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load a pre-trained model (for example purposes, we use Swin Transformer)
    model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    layer_name = args.layer_name

    # A simple hook to capture the features from a specific layer.
    # For demonstration, we assume `hook_specific_layer` is a context manager or function that attaches to the model.
    # Replace this with your hook implementation.
    hook = hook_specific_layer(model, layer_name)

    image_size = args.image_size

    # # Create a Fourier-initialized image.
    # # For simplicity, we create a random image tensor that requires gradients.
    # params = torch.randn(
    #     1, 3, image_size, image_size, device=device, requires_grad=True
    # )
    #
    # def image_f():
    #     return params
    #
    # params, image_f = image()

    params, image_f = image.image(
        image_size, fft=True, decorrelate=False, device=device
    )

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    logging.info("Starting Fourier Feature Visualisation optimisation...")
    fourier_fv(
        model,
        optimizer,
        image_f,
        image_size,
        args.fourier_epochs,
        hook,
        args.neuron_idx,
    )

    # Convert final image tensor to numpy for display or saving.
    final_img = image_f().clone().detach().cpu().numpy().squeeze()
    # Rearrange channels if needed (assumes image in CHW format)
    final_img = np.transpose(final_img, (1, 2, 0))

    if args.save:
        save_image(final_img, "FourierFV", args.output_dir, layer_name, args.neuron_idx)
    if args.no_plot:
        plot_image(final_img, "FourierFV", layer_name, args.neuron_idx)

    hook.close()


# Dummy hook implementation for demonstration.
class SimpleHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def hook_specific_layer(model: torch.nn.Module, layer_name: str):
    """
    Attach a hook to a specific layer.
    This dummy implementation searches the model's named modules.
    """
    for name, module in model.named_modules():
        if name == layer_name:
            logging.info(f"Hook attached to layer: {layer_name}")
            return SimpleHook(module)
    raise ValueError(f"Layer {layer_name} not found in model.")


if __name__ == "__main__":
    main()
