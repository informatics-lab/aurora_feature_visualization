import numpy as np
import matplotlib.pyplot as plt
import torch


def image(
    w,
    h=None,
    sd=None,
    batch=None,
    decorrelate=True,
    fft=True,
    channels=None,
):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    params, image_f = fft_image(shape, sd=sd, device="cpu")
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output


color_correlation_svd_sqrt = np.asarray(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]


def _linear_decorrelate_color(tensor):
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(
        t_permute, torch.tensor(color_correlation_normalized.T).to(t_permute.device)
    )
    tensor = t_permute.permute(0, 3, 1, 2)
    return tensor


def to_valid_rgb(image_f, decorrelate=False):
    def inner():
        image = image_f()
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)

    return inner


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1, device="cpu"):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (
        (batch, channels) + freqs.shape + (2,)
    )  # 2 for imaginary and real components
    sd = sd or 0.01  # sd == standard deviation

    spectrum_real_imag_t = (
        (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    )

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if type(scaled_spectrum_t) is not torch.complex64:
            scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
        image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
        magic = 4.0
        image = image / magic
        return image

    return [spectrum_real_imag_t], inner


# # Display the images
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
# ax[0].imshow(image_array, cmap="gray")
# ax[0].set_title("Original Image")
# ax[0].axis("off")
#
# ax[1].imshow(reconstructed_image, cmap="gray")
# ax[1].set_title("Random Fourier Transform Image")
# ax[1].axis("off")

params, image_f = image(w=1000, channels=1)
# image_array = image_f().squeeze().permute(1, 2, 0).detach()
image_array = image_f().squeeze().detach()
print(image_array.shape)

# Display the generated image
plt.imshow(image_array)
plt.title("Generated Image from Random Magnitude and Phase")
plt.axis("off")
plt.show()
