import torch
import numpy as np


surface_correlation_svd_sqrt = np.asarray(
    [
        [4.4658e02, -1.0148e01, -3.6899e00, 6.2798e03],
        [-1.0148e01, 3.0268e01, -6.0366e-01, -9.4081e02],
        [-3.6899e00, -6.0366e-01, 2.2605e01, -2.2036e01],
        [6.2798e03, -9.4081e02, -2.2036e01, 1.7375e06],
    ]
).astype("float32")

static_correlation_svd_sqrt = np.asarray(
    [
        [6.9711e07, 3.2145e03, 2.4404e03],
        [3.2145e03, 1.3629e00, 4.2424e-01],
        [2.4404e03, 4.2424e-01, 2.1209e-01],
    ]
).astype("float32")

atmospheric_correlation_svd_sqrt = np.asarray(
    [
        [8.5069e02, -1.1403e02, 5.6725e-01, 7.5097e-02, -1.4632e06],
        [-1.1403e02, 2.0265e02, -1.3335e00, -1.2353e-02, 1.8537e05],
        [5.6725e-01, -1.3335e00, 8.7330e01, 1.6846e-04, -3.4764e03],
        [7.5097e-02, -1.2353e-02, 1.6846e-04, 1.2927e-05, -1.1313e02],
        [-1.4632e06, 1.8537e05, -3.4764e03, -1.1313e02, 3.5470e09],
    ]
).astype("float32")


color_mean = [0.48, 0.46, 0.41]


def calculate_correlation_norm(correlation_svd_sqrt):
    max_norm_svd_sqrt = np.max(np.linalg.norm(correlation_svd_sqrt, axis=0))
    return correlation_svd_sqrt / max_norm_svd_sqrt


SURFACE_CORRELATION_NORMALIZED = calculate_correlation_norm(
    surface_correlation_svd_sqrt
)
STATIC_CORRELATION_NORMALIZED = calculate_correlation_norm(static_correlation_svd_sqrt)
ATMOSPHERIC_CORRELATION_NORMALIZED = calculate_correlation_norm(
    atmospheric_correlation_svd_sqrt
)


def _linear_decorrelate_color(tensor, correlation_normalized):
    t_permute = tensor.permute(0, 1, 3, 4, 5, 2)
    t_permute = torch.matmul(
        t_permute, torch.tensor(correlation_normalized.T).to(t_permute.device)
    )
    tensor = t_permute.permute(0, 1, 5, 2, 3, 4)
    return tensor


def to_valid_vars(image_f, decorrelate=False, correlation_normalized=None):
    def inner():
        image = image_f()
        if decorrelate:
            assert (
                correlation_normalized,
                "Correlation Normlaized not defined",
            )
            image = _linear_decorrelate_color(image, correlation_normalized)
        return torch.sigmoid(image)

    return inner
