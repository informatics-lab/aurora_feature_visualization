from fft_image_2 import fft_volume
from image_2 import image

# # Example Usage for 3D Volume:
# shape_3d = (2, 3, 32, 64, 128)  # batch, channels, depth, height, width
# spectrum_3d, generate_volume_3d = fft_volume(shape_3d)
# volume_3d = generate_volume_3d()
# print(f"Generated 3D volume shape: {volume_3d.shape}")
# print(f"Generated 3D volume spectrum shape: {spectrum_3d[0].shape}")


spectrum_3d, generate_volume_3d = image(
    lat=721,
    lon=1440,
    time=2,
    vars=4,
    lvl=1,
    sd=None,
    batch=1,
    decorrelate=True,
    fft=True,
)
volume_3d = generate_volume_3d()
print(f"Generated 3D volume shape: {volume_3d.shape}")
print(f"Generated 3D volume spectrum shape: {spectrum_3d[0].shape}")
