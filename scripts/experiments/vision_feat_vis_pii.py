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
device = torch.device("mps")

model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)

hook = hook_specific_layer(model, "features.1.0.mlp")

image_size = 224
params, image_f = image.image(image_size, fft=True, decorrelate=True, device=device)

transforms = [
    transform.focus(0, 0),  # This is will be updated in the inversion loop
    transform.jitter(8),
    transform.color_jitter_r(1, True),
]

neuron_idx = 0
n_epochs = 400
learning_rate = 5e-2
batch_size = 1

optimizer = torch.optim.Adam(
    params,
    lr=learning_rate,
)


def neuron_loss(tensor, neuron_idx):
    return -tensor[:, :, :, neuron_idx].mean()


def attention_loss(tensor, head_idx, token_idx=None):
    return -tensor[:, head_idx:token_idx].mean()


model.eval()
step = image_size // 8
for i in range(2 * step, image_size + 1, step):
    pbar = trange(n_epochs, desc="loss: -")
    for _ in pbar:
        optimizer.zero_grad()

        transforms[0] = transform.focus(i, 0)
        transform_f = transform.compose(transforms)
        predictions = model(transform_f(image_f()))

        loss = neuron_loss(hook.features, neuron_idx)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():.2f}")

hook.close()

final_image = image_f()
plt.imshow(np.transpose(np.squeeze(final_image.detach().numpy()), (1, 2, 0)))
plt.axis("off")
plt.show()
