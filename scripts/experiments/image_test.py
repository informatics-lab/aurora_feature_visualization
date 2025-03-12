from image import image
import matplotlib.pyplot as plt
import torch
import numpy as np

lat = 721
lon = 1440
time = 2

device = "cpu"

lvl_type = "surf"  # surf / atmos / static

# params, image_f = image(w=1440, h=721, decorrelate=True, channels=3)
params, image_f = image(w=1440, h=721, decorrelate=False, channels=1)

tensor = image_f().squeeze().detach().numpy()
print(tensor.shape)

cols = tensor.shape[0]  # (2, 4)
cols = 2

# Create the figure and axes
fig, axes = plt.subplots(1, cols, figsize=((cols + 1) * 5, 1 * 5))

# Loop through each image and plot
for i in range(cols):
    ax = axes[i]  # Get the correct subplot
    # ax.imshow(tensor[i], cmap="gray")  # Adjust cmap if needed
    ax.imshow(tensor, cmap="gray")  # Adjust cmap if needed
    ax.set_title(f"Slice [{i}]")
    ax.axis("off")  # Hide axes

# ax = axes[cols]  # Get the correct subplot
# ax.imshow(np.transpose(tensor, axes=(1, 2, 0)), cmap="gray")  # Adjust cmap if needed
# ax.set_title(f"Slice [{cols}]")
# ax.axis("off")  # Hide axes

# Adjust layout and show plot
plt.tight_layout()
plt.show()
