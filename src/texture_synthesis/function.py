from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import requests
import numpy as np


STYLE_VECTOR_LENGTH = 3840
STYLE_VECTOR_CONFIG = [
    [64, 64],
    [64, 64],
    [128, 128],
    [128, 128],
    [256, 256],
    [256, 256],
    [256, 256],
    [256, 256],
    [512, 512]
 ]

trans = transforms.Compose(
    [
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

inv_trans = transforms.Compose(
    [
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.5, 1/0.5, 1/0.5 ]),
        transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ], std = [ 1., 1., 1. ]),
    ]
)

def square_crop(image):
    return transforms.CenterCrop(min(image.size))(image)

def image_to_tensor(image, normalize=True):
    result = transforms.ToTensor()(image)
    if normalize:
        return trans(result).unsqueeze(0)
    else:
        return result.unsqueeze(0)

def tensor_to_image(tensor, normalize=True):
    if normalize:
        return transforms.ToPILImage()(inv_trans(tensor[0].clip(-1, 1)))
    else:
        return transforms.ToPILImage()(tensor[0].clip(0, 1))

def get_white_noise_image(w, h):
    pil_map = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return pil_map

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def load_from_url(url: str):
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def get_statistic(features: list, detach=True):
    result = []
    for feature in features:
        b, c, h, w = feature.shape
        feature_reshaped = feature.reshape(c, -1)
    
        mu = torch.mean(feature_reshaped, 1)
        sigma = torch.std(feature_reshaped, 1)

        if detach:
            result += [[mu.detach(), sigma.detach()]]
        else:
            result += [[mu, sigma]]

    return result

def statistic_to_tensor_level(statistic: list):
    result = []
    for mus, sigmas in statistic:
        result += [mus, sigmas]
    return torch.cat(result)

def tensor_to_statistic_level(style_statistic_tensor: torch.Tensor):
    assert len(style_statistic_tensor) == STYLE_VECTOR_LENGTH
    result = []
    current = 0
    for len_mus, len_sigmas in STYLE_VECTOR_CONFIG:
        mus = style_statistic_tensor[current:current+len_mus]
        current = current + len_mus
        sigmas = style_statistic_tensor[current:current+len_sigmas]
        current = current + len_sigmas
        result += [[mus, sigmas]]
    return result

def statistic_to_tensor(statistic: list):
    result = []
    for level in statistic:
        t = statistic_to_tensor_level(level)
        result += [t]
    return torch.cat(result)

def tensor_to_statistic(style_statistic_tensor: torch.Tensor):
    assert len(style_statistic_tensor) % STYLE_VECTOR_LENGTH == 0
    result = []
    current = 0
    for _ in range(len(style_statistic_tensor) // STYLE_VECTOR_LENGTH):
        level_statistic = tensor_to_statistic_level(style_statistic_tensor[current:current+STYLE_VECTOR_LENGTH])
        current += STYLE_VECTOR_LENGTH
        result += [level_statistic]
    return result

def style_loss(st1, st2):
    style_loss = 0
    metric = nn.MSELoss()
    for el_st1, el_st2 in zip(st1, st2):
        mus_st1 = el_st1[0]
        mus_st2 = el_st2[0]
        stds_st1 = el_st1[1]
        stds_st2 = el_st2[1]
        
        style_loss += (metric(mus_st1, mus_st2) + metric(stds_st1, stds_st2)) / len(mus_st1)
    return style_loss

def optimize(base_tensor, style_statistic, feature_extractor, n_iter=50, progress=True):
        if progress:
            iters = tqdm(range(n_iter))
        else:
            iters = range(n_iter)

        base_tensor = base_tensor.requires_grad_(True)
        optimizer = optim.LBFGS([base_tensor], lr=1) 
        
        for i in iters:
            def closure():
                base_statistic = get_statistic(feature_extractor(base_tensor), detach=False)
                total_loss = style_loss(base_statistic, style_statistic)
                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                return total_loss

            optimizer.step(closure)
            if progress:
                iters.set_description(f"Epoch: {i}")

        return base_tensor.detach()
    
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_statistic):
    size = content_feat.size()
    style_mean, style_std = style_statistic
    style_mean = style_mean.reshape(1, -1, 1, 1)
    style_std = style_std.reshape(1, -1, 1, 1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
