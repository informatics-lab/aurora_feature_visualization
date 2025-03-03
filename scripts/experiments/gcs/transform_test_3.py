import torch
import matplotlib.pyplot as plt
from kornia.geometry.transform import warp_affine3d


def jitter_3d(displacement_range, dsize):
    def inner(image_t):
        device = image_t.device
        batch_size = image_t.shape[0]

        dx = (torch.rand(batch_size, 1, device=device) * 2 - 1) * displacement_range
        dy = (torch.rand(batch_size, 1, device=device) * 2 - 1) * displacement_range
        dz = (torch.rand(batch_size, 1, device=device) * 2 - 1) * displacement_range

        M = torch.eye(3, 4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        M[:, 0, 3] = dx.squeeze(1)
        M[:, 1, 3] = dy.squeeze(1)
        M[:, 2, 3] = dz.squeeze(1)

        jittered_image = warp_affine3d(image_t, M, dsize)
        return jittered_image, dx, dy, dz

    return inner


def visualize_3d_translation(original_tensor, jittered_tensor, dx, dy, dz):
    batch_size = original_tensor.shape[0]
    depth, height, width = original_tensor.shape[2:]

    for b in range(batch_size):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Create a grid of points for visualization
        x = torch.arange(width)
        y = torch.arange(height)
        z = torch.arange(depth)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

        # Original points
        ax.scatter(xx, yy, zz, c="blue", marker="o", alpha=0.1, label="Original")

        # Translated points
        translated_x = xx + dx[b].item()
        translated_y = yy + dy[b].item()
        translated_z = zz + dz[b].item()
        ax.scatter(
            translated_x,
            translated_y,
            translated_z,
            c="red",
            marker="o",
            alpha=0.1,
            label="Translated",
        )

        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("Depth")
        ax.set_title(f"3D Translation (Batch {b})")
        ax.legend()
        plt.show()


# Example usage
batch_size = 1
channels = 1
depth = 13
height = 721
width = 1440
displacement_range = 5
dsize = (depth, height, width)

image_tensor = torch.zeros(batch_size, channels, depth, height, width)
image_tensor[0, 0, 5, 5, 5] = (
    1  # Put a single point to make it easier to see translation.
)

jitter_func = jitter_3d(displacement_range, dsize)
jittered_tensor, dx, dy, dz = jitter_func(image_tensor)

# visualize_3d_translation(image_tensor, jittered_tensor, dx, dy, dz)
