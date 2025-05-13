from kornia.geometry.transform import warp_affine
import torch


def jitter_3d(max_shift):
    mode = "bilinear"
    padding_mode = "border"
    align_corners = True

    def inner(
        data: torch.Tensor,
    ) -> torch.Tensor:
        assert data.ndim == 5 and data.shape[0] == 1 and data.shape[2] == 1, (
            "Expected shape (1, T, 1, H, W)"
        )
        B, T, C, H, W = data.shape
        imgs = data.view(B * T, C, H, W)

        shifts = torch.randint(
            low=-max_shift,
            high=max_shift + 1,
            size=(B * T, 2),
            dtype=torch.float,
            device=data.device,
        )

        M = torch.zeros((B * T, 2, 3), device=data.device, dtype=data.dtype)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0
        M[:, :, 2] = shifts  # broadcast tx, ty into last column

        warped = warp_affine(
            imgs,
            M,
            (H, W),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        return warped.view(B, T, C, H, W)

    return inner
