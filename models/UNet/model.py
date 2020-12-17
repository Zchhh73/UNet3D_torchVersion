import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')
from models.UNet import groupnorm


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, interpolate=True, conv_layer_order='crg'):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoders = nn.ModuleList([
            Encoder(in_channels, 64, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(64, 128, conv_layer_order=conv_layer_order),
            Encoder(128, 256, conv_layer_order=conv_layer_order),
            Encoder(256, 512, conv_layer_order=conv_layer_order)
        ])

        self.decoders = nn.ModuleList([
            Decoder(256 + 512, 256, interpolate, conv_layer_order=conv_layer_order),
            Decoder(128 + 256, 128, interpolate, conv_layer_order=conv_layer_order),
            Decoder(64 + 128, 64, interpolate, conv_layer_order=conv_layer_order),
        ])
        self.final_conv = nn.Conv3d(64, out_channels, 1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoders_features in zip(self.decoders, encoders_features):
            x = decoder(encoders_features, x)

        x = self.final_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kerner_size=3, is_max_pool=True, max_pool_kernel_size=(2, 2, 2),
                 conv_layer_order='crg'):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, padding=1) if is_max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=conv_kerner_size, order=conv_layer_order)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate, kernel_size=3, scale_factor=(2, 2, 2),
                 conv_layer_order='crg'):
        super(Decoder, self).__init__()
        if interpolate:
            self.upsample = None
        else:
            self.upsample = nn.ConvTranspose3d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      order=conv_layer_order)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
        else:
            x = self.upsample(x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x


class DoubleConv(nn.Sequential):
    '''
    'cr' -> conv + ReLU
    'crg' -> conv + ReLU + groupnorm
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg'):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        # conv1
        self._add_conv(1, conv1_in_channels, conv1_out_channels,
                       kernel_size, order)
        # conv2
        self._add_conv(2, conv2_in_channels, conv2_out_channels,
                       kernel_size, order)

    def _add_conv(self, pos, in_channels, out_channels, kernel_size, order):
        assert pos in [1, 2], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r' in order, "'r' (ReLU layer) MUST be present"
        assert order[
                   0] is not 'r', 'ReLU cannot be the first operation in the layer'

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module(f'relu{pos}', nn.ReLU(inplace=True))
            elif char == 'c':
                self.add_module(f'conv{pos}', nn.Conv3d(in_channels, out_channels, kernel_size, padding=1))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm3d MUST go after the Conv3d'
                self.add_module(f'norm{pos}', groupnorm.GroupNorm3d(out_channels))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    self.add_module(f'norm{pos}', nn.BatchNorm3d(in_channels))
                else:
                    self.add_module(f'norm{pos}', nn.BatchNorm3d(out_channels))
            else:
                raise ValueError(f"Unsupported layer type '{char}'. MUST be one of 'b', 'r', 'c'")


if __name__ == '__main__':
    model = UNet3D(3, 3, False).cuda()
    model.train()
    x = torch.randn((1, 3, 8, 128, 128)).cuda()
    y = model(x)
    print(y.shape)
