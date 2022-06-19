import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from .net import vgg_sequential
from .function import get_statistic, statistic_to_tensor
import math
import pkg_resources


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()

        last_layer = 31
        state_dict_path = "/".join(("models", "vgg_normalised.pth"))

        self.model = nn.Sequential(*self._load_vgg(state_dict_path)[:last_layer])
        self.required_features = [3, 10, 17, 30]

    def _load_vgg(self, state_dict_path) -> list:
        vgg = vgg_sequential
        resource_package = __name__
        stream = pkg_resources.resource_stream(resource_package, state_dict_path)
        vgg.load_state_dict(torch.load(stream))
        return list(vgg.children())

    @staticmethod
    def _auto_K(image_tensor_wh: list) -> int:
        small_side = min(image_tensor_wh)
        return max(math.ceil(math.log(small_side, 2) - 6), 0)

    @staticmethod
    def _check_K(image_tensor_wh: list, K: int):
        assert min(image_tensor_wh) // 2**K >= 16

    @staticmethod
    def _check_image_tensor(image_tensor: torch.Tensor) -> None:
        assert len(image_tensor.shape) == 4
        b, c, w, h = image_tensor.shape
        assert b == 1
        assert c == 3
        assert min(w, h) >= 16

    def get_style_representation(
        self, image_tensor: torch.Tensor, K=-1
    ) -> torch.Tensor:
        self._check_image_tensor(image_tensor)
        assert isinstance(K, int)
        if K < 0:
            K = self._auto_K(image_tensor.shape[-2:])
        else:
            self._check_K(image_tensor.shape[-2:], K)

        _, _, w, h = image_tensor.shape
        pyramid_levels_statistics = []
        for k in range(K + 1):
            scale = 2**k
            level_image_tensor = F.resize(image_tensor, (w // scale, h // scale))

            with torch.no_grad():
                level_features = self.forward(level_image_tensor)
                pyramid_levels_statistics.append(get_statistic(level_features))

        return statistic_to_tensor(pyramid_levels_statistics[::-1])

    def forward(self, x: torch.Tensor) -> list:
        self._check_image_tensor(x)

        features = []
        for layer_idx, layer in enumerate(self.model):
            x = layer(x)
            if layer_idx in self.required_features:
                features.append(x)
        return features
