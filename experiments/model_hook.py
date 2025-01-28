import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from typing import OrderedDict

checkpoints_dir = "checkpoints"
model_name = "aurora-0.25-small-pretrained.ckpt"
model_path = os.path.join(checkpoints_dir, model_name)

model = AuroraSmall()

model.load_checkpoint_local(model_path)

lat = 180
lon = 360
batch = Batch(
    surf_vars={k: torch.randn(1, 2, lat, lon) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(lat, lon) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, lat, lon) for k in ("z", "u", "v", "t", "q")},
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
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    if return_hooks:
        return hook, features
    return hook

model.eval()
hook, features = hook_model(model, None, return_hooks=True)
prediction = model(batch)

# print(features["encoder"].features)
print(hook("encoder"))
