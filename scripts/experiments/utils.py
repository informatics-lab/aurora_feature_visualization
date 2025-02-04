import torch
import torch.nn as nn

from typing import OrderedDict

# From Lucent
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

# From Plug-in Inversion
class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t
        # x = output_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError
