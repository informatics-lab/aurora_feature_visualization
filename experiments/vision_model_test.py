import torchvision.models as models
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import numpy as np
from typing import OrderedDict
from tqdm import trange
import matplotlib.pyplot as plt
import random

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(pretrained=True)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f, return_hooks=False):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, (
                f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            )
            out = features[layer].features
        assert out is not None, (
            "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        )
        return out

    if return_hooks:
        return hook, features
    return hook


hook, features = hook_model(model, None, return_hooks=True)

x = torch.rand((1, 3, 512, 512), requires_grad=True)

layer_name = "inception4e"
channel_idx = 0
n_epochs = 100
learning_rate = 5e-2

# Define optimizer for the input data
optimizer = torch.optim.Adam(
    [x],
    lr=learning_rate,
)

# Define a list of transformations you want to apply
transformations = [
    T.RandomRotation(degrees=5),
    T.RandomResizedCrop(512, scale=(0.9, 1.1)),
]


def random_translation(image_tensor, max_translation=16):
    dx = random.randint(-max_translation, max_translation)
    dy = random.randint(-max_translation, max_translation)
    translation_matrix = torch.tensor([[1, 0, dx], [0, 1, dy]], dtype=torch.float32)
    grid = F.affine_grid(
        translation_matrix.unsqueeze(0), image_tensor.size(), align_corners=False
    )
    return F.grid_sample(image_tensor, grid, align_corners=False)


def fill_empty_pixels(image_tensor, min_val=-117, max_val=138):
    empty_pixels_mask = image_tensor == 0
    if empty_pixels_mask.any():
        random_values = torch.rand_like(image_tensor) * (max_val - min_val) + min_val
        image_tensor[empty_pixels_mask] = random_values[empty_pixels_mask]
    return image_tensor


def apply_random_transforms(image_tensor):
    random.shuffle(transformations)
    transformed_image = image_tensor
    for transform in transformations:
        transformed_image = transform(transformed_image)
    transformed_image = random_translation(transformed_image)
    transformed_image = fill_empty_pixels(transformed_image)

    return transformed_image


transformed_image = apply_random_transforms(x)

model.train()
pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    optimizer.zero_grad()

    transformed_image = apply_random_transforms(x.detach()).requires_grad_(True)

    predictions = model(x)

    loss = -hook(layer_name)[0, channel_idx, :, :].mean()
    loss.backward(retain_graph=True)

    # loss.backward()
    optimizer.step()

    pbar.set_description(
        f"loss: {loss.item():.2f}"
    )  # Use .item() to extract scalar value


x_min = x.min()
x_max = x.max()
x_scaled = (x - x_min) / (x_max - x_min)

# Then you can visualize it
plt.imshow(np.transpose(np.squeeze(x_scaled.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
