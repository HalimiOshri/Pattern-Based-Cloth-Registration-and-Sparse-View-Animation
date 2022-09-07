import torch
import torch.nn
from subnets.encoder import Encoder

class ClothesAE(torch.nn.Module)
    def __init__(self, conf):
        self.encoder = self.build_encoder(conf.encoder)
        self.decoder = self.build_decoder(conf.decoder)


    def forward(self, normalized_input):
        encoder_out = self.encoder(normalized_input)
        decoder_out = self.decoder(encoder_out["latent"])

        preds = dict()
        return preds

    def build_encoder(self, conf):
        return Encoder(conf)

    def build_decoder(self, conf):
        pass