import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from typing import OrderedDict

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints_dir = "checkpoints"
model_name = "aurora-0.25-small-pretrained.ckpt"
model_path = os.path.join(checkpoints_dir, model_name)

model = AuroraSmall()

model.load_checkpoint_local(model_path)

lat = 180
lon = 360
batch = Batch(
    surf_vars={
        k: torch.randn(1, 2, lat, lon).requires_grad_(True)
        for k in ("2t", "10u", "10v", "msl")
    },
    static_vars={
        k: torch.randn(lat, lon).requires_grad_(True) for k in ("lsm", "z", "slt")
    },
    atmos_vars={
        k: torch.randn(1, 2, 4, lat, lon).requires_grad_(True)
        for k in ("z", "u", "v", "t", "q")
    },
    metadata=Metadata(
        lat=torch.linspace(90, -90, lat),
        lon=torch.linspace(0, 360, lon + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        # This doesn't actually do anything
        self.hook.remove()


def hook_model(model, image_f, return_hooks=False):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
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


# class AdamInput(torch.optim.Adam):
#     pass

n_epochs = 2
learning_rate = 0.05

hook, features = hook_model(model, None, return_hooks=True)

prediction = model(batch)
# print(features["encoder"].features)
print(hook("encoder"))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam([batch.surf_vars["2t"]], lr=learning_rate)

# print(batch.surf_vars["2t"].shape)
# print(batch.surf_vars["2t"].mean())

# print(batch.surf_vars["2t"])
model.train()
for _ in trange(n_epochs):
    # print(batch.surf_vars["2t"])
    predictions = model(batch)

    loss = predictions.surf_vars["2t"].mean()

    # loss = -hook("encoder").mean()
    print(loss)

    loss.backward()  # Calculate gradients

    optimizer.step()  # Update Parameters from gradients
    # optimizer.zero_grad()  # Reset gradients
    # print()

# print(batch.surf_vars["2t"])
print(hook("encoder"))

# print(batch.surf_vars["2t"].grad)

# print(batch.surf_vars["2t"].detach().numpy().shape)

# plt.figure(figsize=(12, 6))  # Adjust figure size if needed
# plt.imshow(
#     batch.surf_vars["2t"][0, 0].detach().numpy(), cmap="coolwarm", origin="lower"
# )  # Use a suitable colormap
# plt.colorbar(label="Temperature (°C)")  # Add a colorbar
# plt.title("T2M Weather Data Visualization")
# plt.xlabel("Longitude Index")
# plt.ylabel("Latitude Index")
#
# # Show the plot
# plt.show()
