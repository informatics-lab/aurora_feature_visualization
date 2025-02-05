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
from augmentations import (
    Clip,
    Jitter,
    Focus,
    RepeatBatch,
    Zoom,
    ColorJitterR,
)
from hooks import hook_specific_layer

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(pretrained=True)

hook = hook_specific_layer(model, "inception4e")

x = torch.rand((1, 3, 512, 512), requires_grad=True)

channel_idx = 0
n_epochs = 100
learning_rate = 5e-2
# learning_rate = 1e-8
batch_size = 1

# Define optimizer for the input data
optimizer = torch.optim.Adam(
    [x],
    lr=learning_rate,
)
loss_fn = nn.CrossEntropyLoss()

model.train()
pbar = trange(n_epochs, desc="loss: -")
for _ in pbar:
    # seq = [
    #     # Focus(x.shape[-1], 1),
    #     Jitter(),
    #     Zoom(),
    #     # RepeatBatch(batch_size),
    #     # ColorJitterR(batch_size, True),
    # ]
    # pre = torch.nn.Sequential(*seq).to(device)
    # x = pre(x.detach()).requires_grad_(True)
    optimizer.zero_grad()

    predictions = model(x)

    # loss = loss_fn(
    #     hook.features[0, channel_idx, :, :],
    #     torch.ones(hook.features.shape)[0, channel_idx, :, :],
    # )

    loss = -hook.features[0, channel_idx, :, :].mean()
    loss.backward(retain_graph=True)

    # loss.backward()
    optimizer.step()

    pbar.set_description(f"loss: {loss.item():.2f}")


x_min = x.min()
x_max = x.max()
x_scaled = (x - x_min) / (x_max - x_min)

# Then you can visualize it
plt.imshow(np.transpose(np.squeeze(x_scaled.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
