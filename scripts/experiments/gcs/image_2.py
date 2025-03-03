from fft_image_2 import fft_volume, pixel_image
from decorrelation import SURFACE_CORRELATION_NORMALIZED, to_valid_vars
import torch


def image(
    lat,
    lon,
    time,
    vars,
    lvl=None,
    sd=None,
    batch=1,
    decorrelate=True,
    fft=True,
    device=torch.device("cpu"),
):
    # shape = [batch, ch, h, w]
    shape = [batch, time, vars, lvl, lat, lon]
    param_f = fft_volume if fft else pixel_image
    params, image_f = param_f(shape, sd=sd, device=device)
    output = to_valid_vars(
        image_f,
        decorrelate=decorrelate,
        correlation_normalized=SURFACE_CORRELATION_NORMALIZED,
    )
    return params, output
