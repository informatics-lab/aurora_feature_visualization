from image import image
import matplotlib.pyplot as plt
import torch
import numpy as np

lat = 721
lon = 1440
time = 2

device = "cpu"

lvl_type = "static"  # using static becuase it is 3 channels that be
lvl_type = "surf"  # surf / atmos / static
lvl_type = "atmos"  # surf / atmos / static

params, image_f = image(lat, lon, time, lvl_type)

tensor = image_f().squeeze().detach().numpy()
tensor = tensor[0, :, 0].squeeze()
print(tensor.shape)

cols = tensor.shape[0]

# Create the figure and axes
fig, axes = plt.subplots(1, cols + 1, figsize=((cols + 1) * 5, 1 * 5))

# Loop through each image and plot
for i in range(cols):
    ax = axes[i]  # Get the correct subplot
    ax.imshow(np.transpose(tensor[i]), cmap="gray")  # Adjust cmap if needed
    ax.set_title(f"Slice [{i}]")
    ax.axis("off")  # Hide axes

ax = axes[cols]  # Get the correct subplot
ax.imshow(np.transpose(tensor, axes=(1, 2, 0)), cmap="gray")  # Adjust cmap if needed
ax.set_title(f"Slice [{cols}]")
ax.axis("off")  # Hide axes

# Adjust layout and show plot
plt.tight_layout()
plt.show()
