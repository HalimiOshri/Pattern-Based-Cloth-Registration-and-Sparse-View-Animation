import torch
import torch.nn as nn
import numpy as np

from models.aux.blocks import (
    Conv2dBias,
    ConvBlock,
    UpConvBlockDeep,
    WeightNorm,
    weights_initializer,
    WarpConv,
)

import logging
logger = logging.getLogger(__name__)

class Decoder(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.uv_size = conf.uv_size
        self.init_uv_size = conf.init_uv_size
        self.num_embs_channels = conf.num_embs_channels
        self.init_channels = conf.init_channels
        self.min_channels = conf.min_channels
        self.lrelu_slope = lrelu_slope

        self.num_blocks = int(np.log2(self.uv_size // self.init_uv_size))
        self.sizes = [self.init_uv_size * 2 ** s for s in range(self.num_blocks + 1)]

        self.num_channels = [
            max(self.init_channels // 2 ** b, self.min_channels)
            for b in range(self.num_blocks + 1)
        ]

        self.resize_tex = nn.UpsamplingBilinear2d((self.uv_size, self.uv_size))

        num_groups = 2  # cloth geometry and texture

        self.conv_block_clothes_verts = ConvBlock(
            self.num_embs_channels, self.init_channels, self.init_uv_size,
        )

        self.conv_block_clothes_tex = ConvBlock(
            self.num_embs_channels, self.init_channels, self.init_uv_size,
        )

        self.embs_resize = nn.UpsamplingBilinear2d(
            size=(self.init_uv_size, self.init_uv_size)
        )

        self.conv_blocks = nn.ModuleList([])
        for b in range(self.num_blocks):
            self.conv_blocks.append(
                UpConvBlockDeep(
                    self.num_channels[b] * num_groups,
                    self.num_channels[b + 1] * num_groups,
                    self.sizes[b + 1],
                    groups=num_groups,
                ),
            )

        self.verts_conv = WeightNorm(
            Conv2dBias(self.num_channels[-1], 3, 3, self.uv_size)
        )

        self.tex_conv = WeightNorm(
            Conv2dBias(self.num_channels[-1], 3, 3, self.uv_size)
        )

        self.apply(weights_initializer(self.lrelu_slope))

        self.clothes_verts_conv.apply(weights_initializer(1.0))
        self.clothes_tex_conv.apply(weights_initializer(1.0))

    def forward(self, input_dictionary):
        pass