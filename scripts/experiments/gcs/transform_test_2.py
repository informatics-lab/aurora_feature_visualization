import torch
import numpy as np
import matplotlib.pyplot as plt
from transform import (
    jitter,
    focus,
    jitter_3d,
    zoom,
    color_jitter_r,
    compose,
)  # Assuming these are adapted

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)

size = 16
z = torch.linspace(0, 1, size)  # Depth
y = torch.linspace(0, 1, size)  # Height
x = torch.linspace(0, 1, size)  # Width
grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")

pattern0 = grid_z
pattern1 = grid_y
pattern2 = grid_x
pattern3 = grid_z * grid_y * grid_x

pattern = torch.stack([pattern0, pattern1, pattern2, pattern3], dim=0)
pattern = pattern.unsqueeze(dim=0)
print(pattern.shape)  # Should be (1, 4, 64, 64, 64)
print(pattern[0, 0])  # Should be (1, 4, 64, 64, 64)

pattern_new = pattern.clone()

batch_size = pattern_new.shape[0]
n_epochs = 1
for epoch in range(n_epochs):
    transforms_list = [
        jitter_3d(5, (size, size, size)),
    ]
    transform_f = compose(transforms_list)
    pattern_new = transform_f(pattern)

print()
print(pattern_new[0, 0])


def normalize_slice(image):
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 0:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = image
    return normalized


def tensor_to_slice(tensor, slice_index=8, channel=0):
    # Extract a slice from the 3D tensor
    image = tensor[
        0, channel, slice_index, :, :
    ]  # Example: slice along the depth dimension
    return image.cpu().detach().numpy()


num_slices = 6  # Number of slices to display
slice_indices = np.linspace(
    0, pattern.shape[2] - 1, num_slices, dtype=int
)  # get even slices

plt.figure(figsize=(12, 6 * num_slices))  # Adjust figure height

for i, slice_index in enumerate(slice_indices):
    orig_slice = tensor_to_slice(pattern, slice_index)
    new_slice = tensor_to_slice(pattern_new, slice_index)

    orig_slice_normalized = normalize_slice(orig_slice)
    new_slice_normalized = normalize_slice(new_slice)

    plt.subplot(num_slices, 2, 2 * i + 1)
    plt.imshow(orig_slice_normalized, interpolation="nearest")
    plt.title(f"Original Slice {slice_index}")
    plt.axis("off")

    plt.subplot(num_slices, 2, 2 * i + 2)
    plt.imshow(new_slice_normalized, interpolation="nearest")
    plt.title(f"New Slice (Augmented) {slice_index}")
    plt.axis("off")

plt.tight_layout()  # prevents overlapping titles.
plt.show()
