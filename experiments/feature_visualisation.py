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

neuron_idx = 1
n_epochs = 100
learning_rate = 0.05
vars = [
    batch.surf_vars["2t"],
    batch.surf_vars["10u"],
    # batch.surf_vars["10v"],
    # batch.surf_vars["msl"],
    # batch.static_vars["lsm"],
    # batch.static_vars["z"],
    # batch.static_vars["slt"],
    # batch.atmos_vars["z"],
    # batch.atmos_vars["u"],
    # batch.atmos_vars["v"],
    # batch.atmos_vars["t"],
    # batch.atmos_vars["q"],
]

# Define optimizer for the input data
optimizer = torch.optim.Adam(
    vars,
    lr=learning_rate,
)

model.eval()
pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    optimizer.zero_grad()
    predictions = model(batch)

    loss = -hook("backbone_encoder_layers_0_blocks_1_mlp_act")[0, :, neuron_idx].mean()
    loss.backward()
    optimizer.step()

    pbar.set_description(f"loss: {loss:.2f}")

rollout_steps = 2
surface_vars = [
    batch.surf_vars["2t"],
    batch.surf_vars["10u"],
    # batch.surf_vars["10v"],
    # batch.surf_vars["msl"],
]
fig, axes = plt.subplots(
    len(surface_vars), rollout_steps, figsize=(15, len(surface_vars) * 5)
)
for i, var_data in enumerate(surface_vars):
    for j in range(rollout_steps):
        ax = axes[i, j]
        im = ax.imshow(var_data[0, j].detach().numpy(), cmap="coolwarm", origin="lower")
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")

plt.tight_layout()
plt.show()

# fig, axes = plt.subplots(3, 2, figsize=(15, 20))
#
# for i, var_data in enumerate(
#     [
#         batch.static_vars["lsm"],
#         batch.static_vars["z"],
#         batch.static_vars["slt"],
#     ]
# ):
#     ax = axes[i]
#     im = ax.imshow(var_data.detach().numpy(), cmap="coolwarm", origin="lower")
#     ax.set_xlabel("Longitude Index")
#     ax.set_ylabel("Latitude Index")
#
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(5, 2, figsize=(15, 20))
#
# for i, var_data in enumerate(
#     [
#         batch.atmos_vars["z"],
#         batch.atmos_vars["u"],
#         batch.atmos_vars["v"],
#         batch.atmos_vars["t"],
#         batch.atmos_vars["q"],
#     ]
# ):
#     for j in range(2):
#         ax = axes[i, j]
#         im = ax.imshow(
#             var_data[0, j, 0].detach().numpy(), cmap="coolwarm", origin="lower"
#         )
#         ax.set_xlabel("Longitude Index")
#         ax.set_ylabel("Latitude Index")
#
# plt.tight_layout()
# plt.show()
