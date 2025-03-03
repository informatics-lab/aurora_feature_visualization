import torch
import numpy as np

TORCH_VERSION = torch.__version__
DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pixel_image(shape, sd=None, device=DEFAULT_DEVICE):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor


def rfftn_freqs(shape):
    """Computes ND spectrum frequencies for real FFT."""
    ndim = len(shape)
    freqs = []
    for i, size in enumerate(shape):
        if i == ndim - 1:  # Last dimension (real FFT)
            if size % 2 == 1:
                freq = np.fft.fftfreq(size)[: size // 2 + 2]
            else:
                freq = np.fft.fftfreq(size)[: size // 2 + 1]
        else:
            freq = np.fft.fftfreq(size)
        freqs.append(freq)

    # Create a meshgrid of frequencies
    mesh = np.meshgrid(*freqs, indexing="ij")

    # Calculate the magnitude of the frequency vector
    return np.sqrt(np.sum(np.array(mesh) ** 2, axis=0))


def fft_volume(shape, sd=None, decay_power=1, device=DEFAULT_DEVICE):
    batch, channels, *spatial_dims = shape
    freqs = rfftn_freqs(spatial_dims)
    init_val_size = (batch, channels) + freqs.shape + (2,)  # 2 for real and imaginary

    sd = sd or 0.01

    spectrum_real_imag_t = (
        (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    )

    max_spatial_dim = max(spatial_dims)
    scale = 1.0 / np.maximum(freqs, 1.0 / max_spatial_dim) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if type(scaled_spectrum_t) is not torch.complex64:
            scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
        volume = torch.fft.irfftn(scaled_spectrum_t, s=spatial_dims, norm="ortho")
        volume = volume[
            :batch, :channels, *([slice(None, dim) for dim in spatial_dims])
        ]
        magic = 4.0
        volume = volume / magic
        return volume

    return [spectrum_real_imag_t], inner
