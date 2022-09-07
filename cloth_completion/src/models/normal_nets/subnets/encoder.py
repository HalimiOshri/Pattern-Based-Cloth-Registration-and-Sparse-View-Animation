import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

from models.aux.blocks import (
    ConvDownBlock,
    WeightNorm,
    weights_initializer,
)

class Encoder(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        # parse conf
        self.uv_size = conf.uv_size
        self.num_blocks = conf.num_blocks
        self.num_embs_channels = conf.num_embs_channels
        self.noise_std = conf.noise_std
        self.encode_texture = conf.encode_texture

        #
        self.sizes = [self.uv_size // 2 ** b for b in range(self.num_blocks)]
        self.resize_tex = nn.UpsamplingBilinear2d((self.uv_size, self.uv_size))

        if isinstance(self.num_channels, int):
            self.num_channels = [3] + self.num_blocks * [self.num_channels]
        else:
            self.num_channels = list(self.num_channels)

        logger.info(f"ConvEncoder: num_channels = {self.num_channels}")

        self.conv_blocks = nn.ModuleList([])

        # NOTE: there are four groups, verts and texture for body and clothes
        if self.encode_texture:
            num_groups = 2
            logger.info('encoding mean texture, num of groups = 4')
        else:
            num_groups = 1
            logger.info('NOT encoding mean texture, num of groups = 2')

        for b in range(self.num_blocks):
            self.conv_blocks.append(
                ConvDownBlock(
                    self.num_channels[b] * num_groups,
                    self.num_channels[b + 1] * num_groups,
                    self.sizes[b],
                    groups=num_groups,
                )
            )

        # TODO: should we put initializer
        self.mu_conv = WeightNorm(
            nn.Conv2d(
                self.num_channels[-1] * num_groups,
                self.num_embs_channels * num_groups,
                1,
                groups=num_groups,
            )
        )
        self.logvar_conv = WeightNorm(
            nn.Conv2d(
                self.num_channels[-1] * num_groups,
                self.num_embs_channels * num_groups,
                1,
                groups=num_groups,
            )
        )

        self.apply(weights_initializer(lrelu_slope))

        logger.warning("NOTE: the initialization for mu / logvar has changed")
        self.mu_conv.apply(weights_initializer(1.0))
        self.logvar_conv.apply(weights_initializer(1.0))

    def forward(self, input_dict):
        verts_normalized_uv = input_dict["verts_normalized_uv"]
        tex_normalized = input_dict["tex_normalized"]
        preds = dict()

        if self.encode_texture:
            joint = torch.cat([verts_normalized_uv, tex_normalized], 1)
        else:
            joint = verts_normalized_uv

        x = joint

        for b in range(self.num_blocks):
            x = self.conv_blocks[b](x)

        # these are body-only embeddings
        embs_mu = self.mu_conv(x)
        embs_logvar = self.logvar_conv(x)

        # NOTE: the noise is only applied to the input-conditioned values
        if self.training:
            noise = torch.randn_like(embs_mu)
            embs = embs_mu + torch.exp(0.5 * embs_logvar) * noise * self.noise_std
        else:
            embs = embs_mu.clone()

        preds.update(
            embs=embs,
            embs_mu=embs_mu,
            embs_logvar=embs_logvar,
        )

        return preds