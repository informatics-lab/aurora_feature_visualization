import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata

checkpoints_dir = "checkpoints"
model_name = "aurora-0.25-small-pretrained.ckpt"
model_path = os.path.join(checkpoints_dir, model_name)

model = AuroraSmall()

model.load_checkpoint_local(model_path)

lat = 720
lon = 1440

batch = Batch(
    surf_vars={k: torch.randn(1, 2, lat, lon) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(lat, lon) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, lat, lon) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 720),
        lon=torch.linspace(0, 360, 1440 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

prediction = model(batch)

print(prediction.surf_vars["2t"].shape)
