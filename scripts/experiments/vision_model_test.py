import torchvision.models as models
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import OrderedDict
from tqdm import trange
import matplotlib.pyplot as plt
import random
from hooks import hook_specific_layer
import image
import transform

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(pretrained=True)

# hook = hook_specific_layer(model, "inception4e")
hook = hook_specific_layer(model, "inception3b")

# x = torch.rand((1, 3, 512, 512), requires_grad=True)
params, image_f = image.image(224, fft=True, decorrelate=True, device=device)

transforms = transform.standard_transforms
transforms.append(transform.preprocess_inceptionv1())
# transforms = transforms.copy()
transform_f = transform.compose(transforms)

channel_idx = 1
n_epochs = 100
learning_rate = 5e-2
batch_size = 1

# Define optimizer for the input data
optimizer = torch.optim.Adam(
    params,
    lr=learning_rate,
)
# loss_fn = nn.CrossEntropyLoss()

model.train()
pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    optimizer.zero_grad()

    predictions = model(transform_f(image_f()))

    # loss = loss_fn(
    #     hook.features[0, channel_idx, :, :],
    #     torch.ones(hook.features.shape)[0, channel_idx, :, :],
    # )

    loss = -hook.features[0, channel_idx, :, :].mean()
    loss.backward(retain_graph=True)

    # loss.backward()
    optimizer.step()

    pbar.set_description(f"loss: {loss.item():.2f}")

x = image_f()

x_min = x.min()
x_max = x.max()
x_scaled = (x - x_min) / (x_max - x_min)

# Then you can visualize it
plt.imshow(np.transpose(np.squeeze(x_scaled.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
