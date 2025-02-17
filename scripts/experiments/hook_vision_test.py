import torch
import torchvision.models as models
from hooks import hook_specific_layer, layer_names
import image

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
# model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
# model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)

params, image_f = image.image(224, fft=True, decorrelate=True, device=device)

print(layer_names(model))

hook = hook_specific_layer(model, "features.3.0.mlp")
model(image_f())

print(hook.features)
