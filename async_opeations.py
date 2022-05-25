from threading import Thread
import torch
from src.texture_synthesis.function import image_to_tensor, tensor_to_image

class ImageLoader(Thread):
    def __init__(self, feature_extractor, pil_images, scale, device):
        super().__init__()
        self.image_tensors = None

        self.feature_extractor = feature_extractor
        self.pil_images = pil_images
        self.scale = scale
        self.device = device

    def run(self):
        self.image_tensors = []
        for pil_image in self.pil_images:
            t = self.feature_extractor.get_style_representation(image_to_tensor(pil_image, False).to(self.device), K=self.scale)
            self.image_tensors.append(t.cpu())
            torch.cuda.empty_cache()


class PreviewGenerator(Thread):
    def __init__(self, SG, style_tensor):
        super().__init__()

        self.generated_texture = None
        self.style_tensor = style_tensor
        self.SG = SG

    def run(self):
        result = self.SG.run(self.style_tensor.to(self.SG.device), [256, 256], 1)
        self.generated_texture = tensor_to_image(result, False)
        torch.cuda.empty_cache()


class Generator(Thread):
    def __init__(self, SG, style_tensor, size, n_iter):
        super().__init__()

        self.generated_texture = None
        self.style_tensor = style_tensor
        self.size = size
        self.n_iter = n_iter
        self.SG = SG

    def run(self):
        result = self.SG.run(self.style_tensor.to(self.SG.device), self.size, self.n_iter)
        self.generated_texture = tensor_to_image(result, False)
        torch.cuda.empty_cache()
