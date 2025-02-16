import torchvision.models as models
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from hooks import hook_specific_layer
import image
import transform

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

hook = hook_specific_layer(model, "inception4e")

params, image_f = image.image(256, fft=True, decorrelate=True, device=device)

transforms = transform.standard_transforms
transforms.append(transform.preprocess_inceptionv1())
transforms = transforms.copy()
transform_f = transform.compose(transforms)

channel_idx = 0
n_epochs = 500
learning_rate = 5e-2
batch_size = 1

optimizer = torch.optim.Adam(
    params,
    lr=learning_rate,
)

model.eval()
pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    optimizer.zero_grad()

    predictions = model(transform_f(image_f()))

    loss = -hook.features[0, channel_idx, :, :].mean()
    loss.backward()
    optimizer.step()

    pbar.set_description(f"loss: {loss.item():.2f}")

final_image = image_f()
plt.imshow(np.transpose(np.squeeze(final_image.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
