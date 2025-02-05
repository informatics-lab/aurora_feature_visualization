import ftt_image
import color
import numpy as np
import matplotlib.pyplot as plt

params, image_f = ftt_image.fft_image((1, 3, 126, 126))
output = color.to_valid_rgb(image_f, decorrelate=False)
x = output()

plt.imshow(np.transpose(np.squeeze(x.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
