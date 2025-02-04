import torch
import matplotlib.pyplot as plt
from augmentations import (
    Clip,
    Jitter,
    Focus,
    RepeatBatch,
    Zoom,
    ColorJitterR,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)

x = torch.linspace(0, 1, 512)
y = torch.linspace(0, 1, 512)
grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

pattern0 = grid_x
pattern1 = grid_y
pattern2 = grid_x * grid_y

pattern = torch.stack([pattern0, pattern1, pattern2], dim=0)
pattern = pattern.unsqueeze(dim=0)
print(pattern.shape)

pattern_new = pattern.clone()

batch_size = pattern_new.shape[0]
n_epochs = 1
for epoch in range(n_epochs):
    seq = [
        Focus(pattern.shape[-1], 1),
        Jitter(),
        RepeatBatch(batch_size),
        ColorJitterR(batch_size, True),
    ]
    pre = torch.nn.Sequential(*seq).to(device)
    pattern_new = pre(pattern_new)


def tensor_to_image(tensor):
    image = tensor[0].permute(1, 2, 0)
    return image.cpu().detach().numpy()


orig_image = tensor_to_image(pattern)
new_image = tensor_to_image(pattern_new)


def normalize_image(image):
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 0:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = image
    return normalized


orig_image_normalized = normalize_image(orig_image)
new_image_normalized = normalize_image(new_image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(orig_image_normalized, interpolation="nearest")
plt.title("Original Pattern")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(new_image_normalized, interpolation="nearest")
plt.title("New Pattern (Augmented)")
plt.axis("off")

plt.show()
