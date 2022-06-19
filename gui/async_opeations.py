from threading import Thread
import torch
from mexts.function import image_to_tensor, tensor_to_image

class CUDAOutOfMemory(Exception):
    def __init__(self, message="CUDA out of memory"):
        self.message = message
        super().__init__(self.message)

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


class Generator(Thread):
    def __init__(self, G, style_tensor, size, n_iter, alpha):
        super().__init__()

        self.generated_texture = None
        self.style_tensor = style_tensor
        self.size = size
        self.n_iter = n_iter
        self.G = G
        self.alpha = alpha
        self.cuda_oom = False

    def run(self):
        result, cuda_oom = self.G.run(self.style_tensor.to(self.G.device), self.size, self.n_iter, self.alpha)
        self.generated_texture = tensor_to_image(result, False)
        torch.cuda.empty_cache()
        if cuda_oom:
            self.cuda_oom = True
            raise CUDAOutOfMemory
