from torch import nn
import torch
import random


class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


class ColorJitter(nn.Module):
    def __init__(
        self,
        batch_size: int,
        shuffle_every: bool = False,
        mean: float = 1.0,
        std: float = 1.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.mean_p = mean
        self.std_p = std
        self.shuffle_every = shuffle_every
        self.mean = None
        self.std = None

    def shuffle(self, device: torch.device):
        # Now the random tensors are created on the device of the input.
        self.mean = (
            (torch.rand((self.batch_size, 3, 1, 1), device=device) - 0.5)
            * 2
            * self.mean_p
        )
        self.std = (
            (torch.rand((self.batch_size, 3, 1, 1), device=device) - 0.5)
            * 2
            * self.std_p
        ).exp()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # If mean/std haven't been created (or if you want new ones every time), generate them
        if self.mean is None or self.shuffle_every:
            self.shuffle(img.device)
        return (img - self.mean) / self.std


class ColorJitterR(ColorJitter):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.shuffle_every:
            self.shuffle(img.device)
        return (img * self.std) + self.mean


class Focus(nn.Module):
    def __init__(self, size: int, std: float):
        super().__init__()
        self.size = size
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Generate the perturbations on the same device as the input.
        pert = (torch.rand(2, device=img.device) * 2 - 1) * self.std
        w, h = img.shape[-2:]
        # Calculate crop positions and ensure they are within bounds.
        x = (pert[0] + w // 2 - self.size // 2).long().clamp(min=0, max=w - self.size)
        y = (pert[1] + h // 2 - self.size // 2).long().clamp(min=0, max=h - self.size)
        return img[:, :, x : x + self.size, y : y + self.size]


class Zoom(nn.Module):
    def __init__(self, out_size: int = 384):
        super().__init__()
        # Create the upsample module without hardcoding a device.
        self.up = nn.Upsample(
            size=(out_size, out_size), mode="bilinear", align_corners=False
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.up(img)


class RepeatBatch(nn.Module):
    def __init__(self, repeat: int = 32):
        super().__init__()
        self.size = repeat

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.repeat(self.size, 1, 1, 1)


class MaskBatch(nn.Module):
    def __init__(self, count: int = -1):
        super().__init__()
        self.count = count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(x[: self.count] if self.count > 0 else x)


class Flip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(3,)) if random.random() < self.p else x
