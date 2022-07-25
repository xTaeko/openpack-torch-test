from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import math
from torchvision import models
from functools import partial
nonlinearity = partial(F.relu,inplace=True)

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DownBlock(nn.Module):
    """A single down-sampling operation for U-Net's encoder.
    Attributes:
        double_conv (nn.Module): -
        pool (nn.MaxPool2d): -
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            pool_size: int = 2,
    ):
        """
        Args:
            in_ch/out_ch (int): input/output channels.
            kernel_size (int): kernel size for convolutions.
            pool_size (int): kernel size of a pooling.
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=(pool_size, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor)
        Returns:
             x, x_xskip:
                 * x (torch.Tensor): encoded tensor.
                 * x_skip (torch.Tensor): tensor to make a skip connection.
        """
        x_skip = self.double_conv(x)
        x = self.pool(x_skip)
        return x, x_skip


class UpBlock(nn.Module):
    """A single upsampling operation for U-Net's encoder.
    Attributes:
        up (nn.Upsampling or nn.ConvTransposed2d): -
        double_conv (DoubleConvBlock): -
    Note:
        ``padding`` is allways set to 'same'.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        """
        Args:
            in_ch (int):
                the number of input channels of ``x1`` (main stream).
            out_ch (int): output channels. Usually, set ``in_ch // 2``.
            pool_size (int): kernel_size for corresponding pooling operation.
        Note:
            ``x2`` (skip connection) should have ``in_ch//`` channels.
        """
        super().__init__()
        # -- Upsamplomg Layer --
        # NOTE: Bilinear Inerpolation with Conv is better than ConvTranspose2d?
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, (1, 3), stride=(1, 2), padding=(0, 1)
        )
        # --  Double Conv Layer --
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                out_ch * 2,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): a tensor from main stream. shape = (N, C, H(=T), W)
            x2 (torch.Tensor): a skip connection tensor from downsampling layer.
                The shape should be (N, C//2, T*2, W).
        Returns:
            torch.Tensor
        """
        assert x1.size(1) == x2.size(1) * 2, f"x1={x1.size()}, x2={x2.size()}"
        assert abs(x1.size(2) - x2.size(2) //
                   2) < 3, f"x1={x1.size()}, x2={x2.size()}"

        # -- upsampling --
        x1 = self.up(x1)

        # -- Concat --
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w //
                        2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x1, x2], dim=1)

        # -- conv --
        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x


# -----------------------------------------------------------------------------

class UNetEncoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            the number of ``DownBlock``.
        pools ([int]):
            list of kernel sizes for pooling.
        conv_blocks (nn.ModuleList): list of ``DownBlock``.
    Todo:
        implement ``get_output_ch(block_index)`` and remove ``filters``.
    """

    def __init__(self, ch_enc: int = 32, depth: int = 5, kernel_size: int = 3):
        super().__init__()
        self.depth = depth

        # -- main blocks --
        input_channels = tuple(
            [ch_enc] + [ch_enc * (2 ** i) for i in range(self.depth - 1)]
        )  # list of input channels.

        blocks = []
        for i, in_ch in enumerate(input_channels):
            if i == 0:
                blocks.append(DownBlock(in_ch, in_ch, pool_size=2))
            else:
                blocks.append(DownBlock(in_ch, in_ch * 2, pool_size=2))
        self.conv_blocks = nn.ModuleList(blocks)

        # -- bottom --
        in_ch = input_channels[-1] * 2
        self.bottom = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch * 2,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(),
            nn.Conv2d(
                in_ch * 2,
                in_ch * 2,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(B,C,T,W)
        Returns:
             encoded, skip_connections
                  * encoded (torch.Tensor): -
                  * skip_connections (list of torch.Tensor): -
        """
        # -- donwnsampling blocks --
        skip_connections = []
        for i in range(self.depth):
            x, x_skip = self.conv_blocks[i](x)
            skip_connections.append(x_skip)

        # -- bottom --
        encoded = self.bottom(x)

        return encoded, skip_connections


class UNetDecoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            the number of ``DownBlock``.
        up_blocks (nn.ModuleList):
            list of ``DownBlock``.
    """

    def __init__(self, ch_enc: int = 32, depth=5):
        """
        Args:
            ch_enc (int): the output channels of the 1st conv block.
            pools ([int]):
                list of kernel sizes for pooling.
        """
        super().__init__()
        self.depth = depth

        # -- main blocks --
        output_channels = tuple(
            reversed([ch_enc * (2 ** i) for i in range(self.depth)])
        )  # list of output channels.

        blocks = []
        for in_ch in output_channels:
            blocks.append(UpBlock(in_ch * 2, in_ch))
        self.up_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor,
                x_skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (Tensor): input
            x_skips ([Tensor]): Output of UTimeEncoder.
        """
        for i in range(self.depth):
            i_inv = (self.depth - 1) - i
            x = self.up_blocks[i](x, x_skips[i_inv])
        return x

# -----------------------------------------------------------------------------


class ResNet(nn.Module):
    """
    Input must take channel-first format (BCHW).
    This model use 2D convolutional filter with kernel size = (f x 1).
    See also original U-net paper at http://arxiv.org/abs/1505.04597
    Note:
        Time axis should come in the 3rd dimention (i.e., H).
    """

    def __init__(
        self,
        in_ch: int = 6,
        num_classes: int = None,
        ch_inc: int = 32,
        depth: int = 5,
    ):
        """
        Args:
            in_ch (int): -
            num_classes (int): The number of classes to model.
            ch_inc (int, optional):
                the number of input channels for UNetEncoder. (Default: 32)
            pools (tuple of int):
               list of kernel sizes for pooling operations.
            depth (int): the number of blocks for Encoder/Decoder.
        """
        super().__init__()

        # NOTE: Add input encoding layer (UNet)
        # Ref:
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        self.inc = nn.Sequential(
            nn.Conv2d(
                in_ch,
                ch_inc,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            nn.BatchNorm2d(ch_inc),
            nn.ReLU(),
        )
        self.encoder = UNetEncoder(ch_inc, depth=depth)
        self.decoder = UNetDecoder(ch_inc, depth=depth)
        #self.dense_clf = nn.Conv2d(ch_inc, num_classes, 1, padding=0, stride=1)

        # Encoder
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        #self.firstconv = resnet.conv1
        self.firstconv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Attention
        # cbam_block, eca_block, se_block
        self.feat1_att = se_block(filters[0])
        self.feat2_att = se_block(filters[1])
        self.feat3_att = se_block(filters[2])
        
        # Center
        self.dblock = SpatialAttention()

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.inc(x)
        #(x, res) = self.encoder(x)
        #x = self.decoder(x, res)
        #x = self.dense_clf(x)

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Attention
        e1 = self.feat1_att(e1)
        e2 = self.feat2_att(e2)
        e3 = self.feat3_att(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
