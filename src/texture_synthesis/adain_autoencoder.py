import torch
import torch.nn as nn
from .net import vgg_sequential, decoder_sequential 
from .function import adaptive_instance_normalization
import pkg_resources

class AdaINAutoencoder(nn.Module):
    def __init__(self) -> None:
        super(AdaINAutoencoder, self).__init__()

        encoder_state_dict_path = '/'.join(('models', 'vgg_normalised.pth'))
        decoder_state_dict_path = '/'.join(('models', 'decoder.pth'))
        encoder_last_layer = 31

        self.encoder = self._load_encoder(encoder_state_dict_path, encoder_last_layer)
        self.decoder = self._load_decoder(decoder_state_dict_path)

    def _load_encoder(self, state_dict_path, last_layer) -> nn.Sequential:
        encoder = vgg_sequential
        resource_package = __name__
        stream = pkg_resources.resource_stream(resource_package, state_dict_path)
        encoder.load_state_dict(torch.load(stream))
        return nn.Sequential(*list(encoder.children())[:last_layer])

    def _load_decoder(self, state_dict_path) -> nn.Sequential:
        decoder = decoder_sequential
        resource_package = __name__
        stream = pkg_resources.resource_stream(resource_package, state_dict_path)
        decoder.load_state_dict(torch.load(stream))
        return decoder

    def forward(self, x, style_statistic, alpha=1):
        assert 0 <= alpha <= 1
        with torch.no_grad():
            content_feat = self.encoder(x)
            encoded = adaptive_instance_normalization(content_feat, style_statistic)
            result = self.decoder(content_feat * (1 - alpha) + encoded * alpha)
        return result
