from typing import OrderedDict

class ModuleHook:
    def __init__(self, module):
        self.features = None
        # Register the hook which will save the output during the forward pass.
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f, return_hooks=False):
    """
    This function attaches forward hooks to all modules in the given model.
    image_f is a callable that should return the input to the model when the hook 'input'
    is requested.
    
    The hook function returned allows you to retrieve the features associated
    with a given layer name. Additionally, if return_hooks is True, the dictionary
    mapping layer names to hooks is also returned.
    """
    # Using model.named_modules() already returns the hierarchical name of every module.
    features = OrderedDict()
    # model.named_modules() includes the top-level model as "" so we skip that.
    for name, module in model.named_modules():
        if name:  # skip the top module if desired
            features[name] = ModuleHook(module)

    def hook(layer):
        if layer == "input":
            # For the 'input' key we call the provided image_f function.
            out = image_f()
        elif layer == "labels":
            # Instead of doing list(features.values())[-1], use next(reversed(...))
            try:
                out = next(reversed(features.values())).features
            except StopIteration:
                raise RuntimeError("No layers have been hooked, so no 'labels' feature exists.")
        else:
            if layer not in features:
                valid_layers = ", ".join(features.keys())
                raise ValueError(
                    f"Invalid layer '{layer}'. Valid layers are: {valid_layers}"
                )
            out = features[layer].features

        if out is None:
            raise RuntimeError(
                "No saved feature maps were found. Ensure the model is in eval mode and the forward pass has been made."
            )
        return out

    if return_hooks:
        return hook, features
    return hook
