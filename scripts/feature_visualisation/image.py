from fft_image import fft_volume, pixel_image
from decorrelation import (
    to_valid_vars,
    SURFACE_CORRELATION_NORMALIZED,
    ATMOSPHERIC_CORRELATION_NORMALIZED,
    STATIC_CORRELATION_NORMALIZED,
)
import torch


correlation_matrix_dict = {
    "surf": SURFACE_CORRELATION_NORMALIZED,
    "atmos": ATMOSPHERIC_CORRELATION_NORMALIZED,
    "static": STATIC_CORRELATION_NORMALIZED,
}

vars_dict = {
    "surf": 4,
    "atmos": 5,
    "static": 3,
}

lvl_dict = {
    "surf": 1,
    # "atmos": 13,
    "atmos": 4,
    "static": 1,
}


def image(
    lat,
    lon,
    time,
    lvl_type=None,
    vars=None,
    lvl=None,
    correlation_matrix=None,
    sd=None,
    batch=1,
    decorrelate=True,
    fft=True,
    device=torch.device("cpu"),
):
    if lvl_type and not (vars and lvl and correlation_matrix):
        correlation_matrix = correlation_matrix_dict[lvl_type]
        vars = vars_dict[lvl_type]
        lvl = lvl_dict[lvl_type]

    shape = [batch, time, vars, lvl, lat, lon]
    param_f = fft_volume if fft else pixel_image
    params, image_f = param_f(shape, sd=sd, device=device)

    output = to_valid_vars(
        image_f,
        decorrelate=decorrelate,
        correlation_normalized=correlation_matrix,
    )
    return params, output
