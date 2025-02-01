import os
import torch
from datetime import datetime
from aurora import AuroraSmall, Batch, Metadata
import matplotlib.pyplot as plt
from tqdm import trange

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

n_epochs = 20
learning_rate = 0.05

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

for _ in trange(n_epochs):
    optimizer.zero_grad()  # Zero gradients

    predictions = model(batch)

    # Define a loss function based on your task
    loss = predictions.surf_vars["2t"].mean()

    loss.backward()  # Calculate gradients

    optimizer.step()  # Update Parameters from gradients

# # Visualize the optimized input data
# plt.figure(figsize=(12, 6))
# plt.imshow(
#     batch.surf_vars["2t"][0, 0].detach().numpy(), cmap="coolwarm", origin="lower"
# )
# plt.colorbar(label="Temperature (°C)")
# plt.title("T2M Weather Data Visualization")
# plt.xlabel("Longitude Index")
# plt.ylabel("Latitude Index")
# plt.show()

# Visualize the original and optimized input data
plt.figure(figsize=(12, 6))

# Original data
plt.subplot(1, 2, 1)
plt.imshow(original_2t[0, 0].numpy(), cmap="coolwarm", origin="lower")
plt.colorbar(label="Temperature (°C)")
plt.title("Original T2M Weather Data")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")

# Optimized data
plt.subplot(1, 2, 2)
plt.imshow(
    batch.surf_vars["2t"][0, 0].detach().numpy(), cmap="coolwarm", origin="lower"
)
plt.colorbar(label="Temperature (°C)")
plt.title("Optimized T2M Weather Data")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")

plt.tight_layout()
plt.show()
