import torchvision.models as models
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from hooks import hook_specific_layer
import image
import transform


def neuron_loss(tensor, neuron_idx):
    return -tensor[:, :, :, neuron_idx].mean()


def attention_loss(tensor, head_idx):
    return -tensor[:, head_idx].mean()


def plugin_in_inversion(
    model, optimizer, image_f, image_size, n_epochs, hook, neuron_idx
):
    transforms = [
        transform.focus(0, 0),  # This is will be updated in the inversion loop
        transform.jitter(8),
        transform.color_jitter_r(1, True),
    ]

    model.eval()
    step = image_size // 8
    for i in range(2 * step, image_size + 1, step):
        cumulative_loss = 0
        pbar = trange(n_epochs, desc="loss: -")
        for epoch in pbar:
            optimizer.zero_grad()

            transforms[0] = transform.focus(i, 0)
            transform_f = transform.compose(transforms)
            _ = model(transform_f(image_f()))

            loss = neuron_loss(hook.features, neuron_idx)
            cumulative_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"loss: {cumulative_loss / (epoch + 1):.3f}")


def fourier_fv(model, optimizer, image_f, image_size, n_epochs, hook, neuron_idx):
    transforms = [
        transform.jitter(8),
        transform.random_scale_vit(
            [1 + (i - 5) / 50.0 for i in range(11)],
            target_size=(image_size, image_size),
        ),
        transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
        transform.jitter(4),
    ]

    transform_f = transform.compose(transforms)

    cumulative_loss = 0
    pbar = trange(n_epochs, desc="loss: -")
    for epoch in pbar:
        optimizer.zero_grad()

        _ = model(transform_f(image_f()))

        loss = neuron_loss(hook.features, neuron_idx)
        cumulative_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {cumulative_loss / (epoch + 1):.3f}")


def main():
    seed = 0
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1).to(device)

    layer_name = "features.3.0.mlp.2"
    hook = hook_specific_layer(model, layer_name)

    image_size = 224

    params, image_f = image.image(image_size, fft=True, decorrelate=True, device=device)

    neuron_idx = 11
    learning_rate = 5e-2

    optimizer = torch.optim.Adam(
        params,
        lr=learning_rate,
    )

    fourier_fv_n_epochs = 500
    fourier_fv(
        model, optimizer, image_f, image_size, fourier_fv_n_epochs, hook, neuron_idx
    )

    first_stage_image = image_f().clone().detach().cpu().numpy()
    first_stage_image = np.transpose(np.squeeze(first_stage_image), (1, 2, 0))

    params, image_f = image.image(image_size, fft=True, decorrelate=True, device=device)

    optimizer = torch.optim.Adam(
        params,
        lr=learning_rate,
    )

    pii_n_epochs = 200
    plugin_in_inversion(
        model, optimizer, image_f, image_size, pii_n_epochs, hook, neuron_idx
    )

    second_stage_image = image_f().clone().detach().cpu().numpy()
    second_stage_image = np.transpose(np.squeeze(second_stage_image), (1, 2, 0))

    pii_fourier_fv_n_epochs = 500
    fourier_fv(
        model, optimizer, image_f, image_size, pii_fourier_fv_n_epochs, hook, neuron_idx
    )

    hook.close()

    # Save image after second stage of training
    third_stage_image = image_f().detach().cpu().numpy()
    third_stage_image = np.transpose(np.squeeze(third_stage_image), (1, 2, 0))

    # Plot both images side by side
    plt.figure(figsize=(10, 5))
    plt.title(
        f"Feature Visualisation: layer - {layer_name}, neuron index - {neuron_idx}"
    )
    plt.tight_layout()
    plt.axis("off")

    plt.subplot(1, 3, 1)
    plt.imshow(first_stage_image)
    plt.title("FourierFV")
    plt.axis("off")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.imshow(second_stage_image)
    plt.title("PII")
    plt.axis("off")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.imshow(third_stage_image)
    plt.title("PII + FourierFV")
    plt.axis("off")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
