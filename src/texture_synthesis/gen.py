from .feature_extractor import FeatureExtractor
import torchvision.transforms.functional as F
from .function import optimize, tensor_to_statistic
from .adain_autoencoder import AdaINAutoencoder
import torch
import torch.nn as nn
from PIL import Image

class SlowGen():
    def __init__(self, feature_extractor: FeatureExtractor, device) -> None:
        self.feature_extractor = feature_extractor
        self.device = device

    def run(self, style_tensor: torch, size: list, n_iter: int) -> torch.Tensor:
        style_statistic = tensor_to_statistic(style_tensor)
        K = len(style_statistic) - 1
        FeatureExtractor._check_K(size, K)
        cuda_oom = False

        w, h = size
        w_base = w // 2 ** K
        h_base = h // 2 ** K

        base_tensor = torch.rand(1, 3, w_base, h_base).to(self.device)

        for k in range(K+1):
            if k == K:
                w_level = w
                h_level = h
            else:
                w_level = w_base * 2 ** k
                h_level = w_base * 2 ** k
                
            print((w_level, h_level))
            base_tensor = F.resize(base_tensor, (w_level, h_level))

            try:
                base_tensor = optimize(base_tensor, style_statistic[k], self.feature_extractor, n_iter)
            except Exception as ex:
                if "out of memory" in str(ex):
                    torch.cuda.empty_cache()
                    cuda_oom = True
                    return base_tensor, cuda_oom

        return base_tensor, cuda_oom


class FastGen2():
    def __init__(self, feature_extractor: FeatureExtractor, autoencoder: AdaINAutoencoder, device) -> None:
        self.feature_extractor = feature_extractor
        self.autoencoder = autoencoder
        self.device = device

    def run(self, style_tensor: torch, size: list, n_iter=1, alpha=0) -> Image:
        style_statistic = tensor_to_statistic(style_tensor)
        K = len(style_statistic) - 1
        FeatureExtractor._check_K(size, K)
        cuda_oom = False

        w, h = size
        w_base = w // 2 ** K
        h_base = h // 2 ** K

        base_tensor = torch.rand(1, 3, w_base, h_base).to(self.device)

        for k in range(K+1):
            if k == K:
                w_level = w
                h_level = h
            else:
                w_level = w_base * 2 ** k
                h_level = w_base * 2 ** k
                
            print((w_level, h_level))
            base_tensor = F.resize(base_tensor, (w_level, h_level))
            
            try:
                if alpha != 0 and k != 0:
                    base_tensor = self.autoencoder(base_tensor, style_statistic[k][-1])
                if n_iter != 0:
                    base_tensor = optimize(base_tensor, style_statistic[k], self.feature_extractor, n_iter)
            except Exception as ex:
                if "out of memory" in str(ex):
                    torch.cuda.empty_cache()
                    cuda_oom = True
                    return base_tensor, cuda_oom

        return base_tensor, cuda_oom
