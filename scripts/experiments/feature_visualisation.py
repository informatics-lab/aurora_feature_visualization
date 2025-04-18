import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
import matplotlib.pyplot as plt
from tqdm import trange
from hooks import hook_specific_layer
import image

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints_dir = "checkpoints"
model_name = "aurora-0.25-small-pretrained.ckpt"
model_path = os.path.join(checkpoints_dir, model_name)

model = AuroraSmall()
model.load_checkpoint_local(model_path)

# Move the model to the chosen device and set it to evaluation mode.
model.to(device).eval()

hook = hook_specific_layer(model, "backbone.encoder_layers.0.blocks.1.mlp.act")

lat = 180
lon = 360


def build_era_image():
    param, image_f = image.image(w=lon, h=lat, batch=1, channels=5, decorrelate=True)


# param, image_f = image.image(shape=[1, 2, lat, lon], decorrelate=True)
param, image_f = image.image(w=lon, h=lat, batch=1, channels=5, decorrelate=True)
print(image_f().shape)
input()
params = [param]

surf_images = {
    k: image.image(shape=[1, 2, lat, lon], decorrelate=True)
    for k in ("2t", "10u", "10v", "msl")
}
static_images = {
    k: image.image(shape=[lat, lon], decorrelate=True) for k in ("lsm", "z", "slt")
}

atmos_images = {
    k: image.image(shape=[1, 2, 4, lat, lon], decorrelate=True)
    for k in ("z", "u", "v", "t", "q")
}

surf_vars = {k: surf_images[k][1]() for k in ("2t", "10u", "10v", "msl")}
static_vars = {k: static_images[k][1]() for k in ("lsm", "z", "slt")}
atmos_vars = {k: atmos_images[k][1]() for k in ("z", "u", "v", "t", "q")}

# Create the batch data on the CPU first.
batch = Batch(
    surf_vars=surf_vars,
    static_vars=static_vars,
    atmos_vars=atmos_vars,
    metadata=Metadata(
        lat=torch.linspace(90, -90, lat, device=device),
        lon=torch.linspace(0, 360, lon + 1, device=device)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

print(batch)
input()


# # Move all batch variables to the chosen device.
# def move_batch_to_device(batch_obj, device):
#     for key in batch_obj.surf_vars:
#         batch_obj.surf_vars[key] = batch_obj.surf_vars[key].to(device)
#     for key in batch_obj.static_vars:
#         batch_obj.static_vars[key] = batch_obj.static_vars[key].to(device)
#     for key in batch_obj.atmos_vars:
#         batch_obj.atmos_vars[key] = batch_obj.atmos_vars[key].to(device)
#     # If Metadata has tensors and you need them on the device, move them too.
#     if hasattr(batch_obj.metadata, "lat"):
#         batch_obj.metadata.lat = batch_obj.metadata.lat.to(device)
#     if hasattr(batch_obj.metadata, "lon"):
#         batch_obj.metadata.lon = batch_obj.metadata.lon.to(device)
#
#
# move_batch_to_device(batch, device)

neuron_idx = 1
n_epochs = 100
learning_rate = 1e-8
vars = [
    batch.surf_vars["2t"],
]


# for var in vars:
#     params.append(fft_image.fft_image(var))


def neuron_loss(tensor: torch.Tensor, neuron_idx: int) -> torch.Tensor:
    return -tensor[:, :, :, neuron_idx].mean()


# Define optimizer for the input data
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.5, 0.99), eps=1e-8)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 0.0)

pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    optimizer.zero_grad()
    predictions = model(batch)

    # Calculate loss from one of the hooked layers.
    print(hook.features.shape)
    input()

    loss = -hook.features[:, :, neuron_idx].mean()

    # loss = neuron_loss(hook.features, neuron_idx)

    loss.backward()
    optimizer.step()
    # lr_scheduler.step()

    pbar.set_description(f"loss: {loss:.2f}")

rollout_steps = 2
surface_vars = [
    batch.surf_vars["2t"],
    batch.surf_vars["10u"],
    batch.surf_vars["10v"],
    batch.surf_vars["msl"],
]
fig, axes = plt.subplots(
    len(surface_vars), rollout_steps, figsize=(15, len(surface_vars) * 5)
)
for i, var_data in enumerate(surface_vars):
    for j in range(rollout_steps):
        ax = axes[i, j]
        # Make sure to transfer the data back to CPU for plotting if needed.
        im = ax.imshow(
            var_data[0, j].detach().cpu().numpy(), cmap="coolwarm", origin="lower"
        )
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")

plt.tight_layout()
plt.show()
