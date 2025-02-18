import torch
import numpy as np

TORCH_VERSION = torch.__version__
DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pixel_image(shape, sd=None, device=DEFAULT_DEVICE):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor


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


def fft_image(shape, sd=None, decay_power=1, device=DEFAULT_DEVICE):
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
        image = image[:batch, :channels, :h, :w]
        magic = 4.0
        image = image / magic
        return image

    return [spectrum_real_imag_t], inner


# Split into the magnitude and phase components of the fft
def fft_image_split(shape, sd=None, decay_power=1, device=DEFAULT_DEVICE):
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

        # Compute magnitude and phase
        magnitude = torch.abs(scaled_spectrum_t)
        phase = torch.angle(scaled_spectrum_t)

        image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
        image = image[:batch, :channels, :h, :w]
        magic = 4.0
        image = image / magic
        return image, magnitude, phase

    return [spectrum_real_imag_t], inner
