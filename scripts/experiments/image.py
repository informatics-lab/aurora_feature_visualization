from fft_image import fft_image, pixel_image
from color import to_valid_rgb
import torch


def image(
    w,
    h=None,
    sd=None,
    batch=None,
    decorrelate=True,
    fft=True,
    channels=None,
    device=torch.device("cpu"),
):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd, device=device)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output
