import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
from hooks import hook_specific_layer, layer_names

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints_dir = "checkpoints"
model_name = "aurora-0.25-small-pretrained.ckpt"
model_path = os.path.join(checkpoints_dir, model_name)

# model = AuroraSmall()
model = AuroraSmall(
    use_lora=False,  # Model was not fine-tuned.
    autocast=True,  # Use AMP.
)

# model.load_checkpoint_local(model_path)

model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model.configure_activation_checkpointing()
model.to(device)
model.eval()

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

print(layer_names(model))

# hook = hook_specific_layer(
#     model, "backbone.encoder_layers.0._checkpoint_wrapped_module.blocks.0.mlp.act"
# )
#
# prediction = model(batch)
#
# print(hook.features)
