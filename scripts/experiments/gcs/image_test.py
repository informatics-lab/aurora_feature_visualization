from image import image

lat = 721
lon = 1440
time = 2

lvl_type = "surf"  # surf / atmos / static

params, image_f = image(lat, lon, time, lvl_type)

print(len(params))
print(params[0].shape)
print(params[0].squeeze(axis=3).shape)
print(image_f().shape)
