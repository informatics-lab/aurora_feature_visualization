from collections import OrderedDict


class ModuleHook:
    def __init__(self, module):
        # Save the module and attach the forward hook to capture its output.
        self.module = module
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    """
    Recursively hooks all layers in the model.

    image_f: A callable which produces input to the model (for example, an image loader),
             because in the returned hook function, the special key "input" calls image_f.

    Returns:
      hook: A function that takes a layer name (or special keys "input"/"labels") and returns the features.
      features (optional): A dict mapping layer names (e.g., "conv1_0") to their ModuleHook instances.
    """
    features = OrderedDict()

    # Recursive function to attach hooks on all submodules
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    continue
                full_name = ".".join(prefix + [name])
                features[full_name] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            # Here we take features from the last hooked layer.
            out = list(features.values())[-1].features
        else:
            if layer not in features:
                raise ValueError(
                    f"Invalid layer {layer}. Retrieve the list of hooked layers by examining the keys of the features dictionary."
                )
            out = features[layer].features
        if out is None:
            raise RuntimeError(
                "No feature maps captured. Make sure to run a forward pass "
                "after setting the model to eval mode (model.eval())."
            )
        return out

    return hook


def hook_specific_layer(model, layer_path_str):
    """
    Hooks a specific layer in the model.

    The layer_path_str should be a string with underscore-separated names corresponding
    to the path of submodules. For example, "conv1_0" will try to hook model.conv1._modules["0"].

    Returns:
      A tuple containing:
         - hook_obj: The ModuleHook instance attached to the identified module.
         - get_features: A function which returns the captured features (after a forward pass).
                      If no features are present, an error is raised.
    """
    names = layer_path_str.split(".")
    module = model
    for name in names:
        if not hasattr(module, "_modules") or name not in module._modules:
            raise ValueError(f"Layer '{layer_path_str}' not found in the model.")
        module = module._modules[name]

    hook_obj = ModuleHook(module)

    return hook_obj


def layer_names(model):
    layers = []

    def append_layers(model, prefix=[]):
        if hasattr(model, "_modules"):
            for name, layer in model._modules.items():
                if layer is None:
                    continue
                layers.append(".".join(prefix + [name]))
                append_layers(layer, prefix=prefix + [name])

    append_layers(model)

    return layers
