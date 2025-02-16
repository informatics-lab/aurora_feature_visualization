import fft_image
import color
import numpy as np
import matplotlib.pyplot as plt
import transform


# We optimise the complex params output
params, image_f = fft_image.fft_image((1, 3, 512, 512))

output = color.to_valid_rgb(image_f, decorrelate=False)
x = output()

test = transform.normalize()
x = test(x)

print(x)

plt.imshow(np.transpose(np.squeeze(x.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
