from melts.feature_extractor import FeatureExtractor
from melts.adain_autoencoder import AdaINAutoencoder
from melts.function import *
from melts.gen import TextureGen
import torch

def main():
    device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
    FE = FeatureExtractor().to(device)
    AA = AdaINAutoencoder().to(device)

    image = load_from_url("https://s3.envato.com/files/269276611/Green%20leaves%20with%20texture.jpg").resize((256, 256))
    t = FE.get_style_representation(image_to_tensor(image).to(device), K=2)
    t = t.detach()

    TG = TextureGen(FE, AA, device)
    tensor_to_image(TG.run(t, [256, 256], 1)[0]).save("test.png")
    tensor_to_image(TG.run(t, [256, 256], 0, 1)[0]).save("test_autoencoder_only.png")

if __name__=='__main__':
    main()