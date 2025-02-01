import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
import matplotlib.pyplot as plt
from tqdm import trange
from typing import OrderedDict

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints_dir = "checkpoints"
model_name = "aurora-0.25-small-pretrained.ckpt"
model_path = os.path.join(checkpoints_dir, model_name)

model = AuroraSmall()

model.load_checkpoint_local(model_path)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input_placeholder, output):
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


hook, features = hook_model(model, None, return_hooks=True)

lat = 180
lon = 360
batch = Batch(
    surf_vars={
        k: torch.randn(1, 2, lat, lon, requires_grad=True)
        for k in ("2t", "10u", "10v", "msl")
    },
    static_vars={
        k: torch.randn(lat, lon, requires_grad=True) for k in ("lsm", "z", "slt")
    },
    atmos_vars={
        k: torch.randn(1, 2, 4, lat, lon, requires_grad=True)
        for k in ("z", "u", "v", "t", "q")
    },
    metadata=Metadata(
        lat=torch.linspace(90, -90, lat),
        lon=torch.linspace(0, 360, lon + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

# Store the original data for comparison
original_2t = batch.surf_vars["2t"].clone().detach()

n_epochs = 10
learning_rate = 0.5

# Define optimizer for the input data
optimizer = torch.optim.Adam(
    [
        batch.surf_vars["2t"],
        batch.surf_vars["10u"],
        batch.surf_vars["10v"],
        batch.surf_vars["msl"],
        batch.static_vars["lsm"],
        batch.static_vars["z"],
        batch.static_vars["slt"],
        batch.atmos_vars["z"],
        batch.atmos_vars["u"],
        batch.atmos_vars["v"],
        batch.atmos_vars["t"],
        batch.atmos_vars["q"],
    ],
    lr=learning_rate,
)

model.eval()  # Set model to evaluation mode

pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    optimizer.zero_grad()  # Zero gradients

    predictions = model(batch)

    # Define a loss function based on your task
    # loss = predictions.surf_vars["2t"].mean()
    loss = -hook("encoder").mean()

    loss.backward()  # Calculate gradients

    optimizer.step()  # Update Parameters from gradients
    pbar.set_description(f"loss: {loss:.2f}")

# Visualize the optimized input data
plt.figure(figsize=(12, 6))
plt.imshow(
    batch.surf_vars["2t"][0, 0].detach().numpy(), cmap="coolwarm", origin="lower"
)
plt.colorbar(label="Temperature (°C)")
plt.title("T2M Weather Data Visualization")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.show()
