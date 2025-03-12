from image import image
import matplotlib.pyplot as plt
import torch
import numpy as np
from decorrelation import (
    to_valid_vars,
    SURFACE_CORRELATION_NORMALIZED,
    ATMOSPHERIC_CORRELATION_NORMALIZED,
    STATIC_CORRELATION_NORMALIZED,
)

lat = 721
lon = 1440
time = 2

device = "cpu"

lvl_type = "surf"  # surf / atmos / static

params, image_f = image(lat, lon, time, lvl_type)

tensor = image_f().squeeze().detach().numpy()


data = np.random.rand(2, 1, 721, 1440, 4)
# fft_spatial = np.fft.fft2(data, axes=(2, 3))  # Apply 2D FFT to lat-lon (Y, X)
tensor = np.fft.rfftn(data, axes=(0, 1, 2, 3))
# Fourier Transform along the time dimension
# fft_time = np.fft.fft(data, axis=0)  # Apply FFT along the time axis (T)


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


def fft_image(shape, sd=None, decay_power=1, device=None):
    batch, time, channels, pl, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (
        (batch, time, channels, pl) + freqs.shape + (2,)
    )  # 2 for imaginary and real components
    sd = sd or 0.01  # sd == standard deviation

    spectrum_real_imag_t = (
        (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    )

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if type(scaled_spectrum_t) is not torch.complex64:
            scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
        image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
        magic = 4.0
        image = image / magic
        return image

    return [spectrum_real_imag_t], inner


shape = [1, 2, 4, 1, 721, 1440]
params, image_f_i = fft_image(shape, device=device)

image_f = to_valid_vars(
    image_f_i,
    decorrelate=True,
    correlation_normalized=SURFACE_CORRELATION_NORMALIZED,
)

print(image_f().shape)

tensor = image_f().squeeze().detach().numpy()

# Define the number of rows and columns for subplots
rows, cols = tensor.shape[:2]  # (2, 4)

# Create the figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

# Loop through each image and plot
for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]  # Get the correct subplot
        ax.imshow(tensor[i, j], cmap="gray")  # Adjust cmap if needed
        ax.set_title(f"Slice [{i}, {j}]")
        ax.axis("off")  # Hide axes

# Adjust layout and show plot
plt.tight_layout()
plt.show()
