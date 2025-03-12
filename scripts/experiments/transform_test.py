import torch
import matplotlib.pyplot as plt
from transform import jitter, focus, zoom, color_jitter_r, compose

if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)

y = torch.linspace(0, 1, 360)
x = torch.linspace(0, 1, 360)
grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

pattern0 = grid_y
pattern1 = grid_x
pattern2 = grid_y * grid_x

pattern = torch.stack([pattern0, pattern1, pattern2], dim=0)
pattern = pattern.unsqueeze(dim=0)
print(pattern.shape)

pattern_new = pattern.clone()

batch_size = pattern_new.shape[0]
n_epochs = 6
for epoch in range(n_epochs):
    transforms_list = [
        focus(int((epoch + 1 / 5) * 360), 0),
        # zoom(720),
        # jitter(8),
        # color_jitter_r(1, True),
        # transform.random_scale_vit(
        #     [1 + (i - 5) / 50.0 for i in range(11)],
        #     target_size=(image_size, image_size),
        # ),
        # transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
        # transform.jitter(4),
    ]
    transform_f = compose(transforms_list)
    pattern_new = transform_f(pattern)

    print(pattern_new.shape)


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
