""" Full assembly of the parts to form the complete network """
import torch
from select import select
from .unet_parts import *
import einops
from .attention import *
from .utils import init_modules, predict_depth, post_process_depth, adaptative_cat
from .submodules import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, model_factor=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channel_list[3], channel_list[4] // factor)
        if self.with_ca is True:
            self.ca = ChannelwiseAttention(channel_list[4] // factor)
        self.up1 = Up(channel_list[4], channel_list[3] // factor, bilinear)
        self.up2 = Up(channel_list[3], channel_list[2] // factor, bilinear)
        self.up3 = Up(channel_list[2], channel_list[1] // factor, bilinear)
        self.up4 = Up(channel_list[1], channel_list[0], bilinear)
        self.outc = OutConv(channel_list[0], self.n_classes)
        init_modules(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.with_ca is True:
            x5_ca, _ = self.ca(x5)
            x5 = torch.mul(x5, x5_ca)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = torch.sigmoid(self.outc(x))
        return logits

class UNet_CA_SA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, with_sa=False, model_factor=1):
        super(UNet_CA_SA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.with_sa = with_sa
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channel_list[3], channel_list[4] // factor)
        if self.with_ca is True:
            self.ca = ChannelwiseAttention(channel_list[4] // factor)
        self.sa0 = SpatialAttention(channel_list[4] // factor)
        self.outc0= OutConv(channel_list[4] // factor, self.n_classes)
        self.up1 = Up(channel_list[4], channel_list[3] // factor, bilinear)
        self.sa1 = SpatialAttention(channel_list[3] // factor)
        self.outc1= OutConv(channel_list[3] // factor, self.n_classes)
        self.up2 = Up(channel_list[3], channel_list[2] // factor, bilinear)
        self.sa2 = SpatialAttention(channel_list[2] // factor)
        self.outc2 = OutConv(channel_list[2] // factor, self.n_classes)
        self.up3 = Up(channel_list[2], channel_list[1] // factor, bilinear)
        self.sa3 = SpatialAttention(channel_list[1] // factor)
        self.outc3 = OutConv(channel_list[1] // factor, self.n_classes)
        self.up4 = Up(channel_list[1], channel_list[0], bilinear)
        self.sa4 = SpatialAttention(channel_list[0] // factor)
        self.outc4 = OutConv(channel_list[0] // factor, self.n_classes)

        init_modules(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.with_ca is True:
            x5_ca, _ = self.ca(x5)
            x5 = torch.mul(x5, x5_ca)
        if self.with_sa is True:
            x_sa = self.sa0(x5)
            x5 = torch.mul(x5, x_sa)
        logits_0 = torch.sigmoid(self.outc0(x5))
        x = self.up1(x5, x4)
        if self.with_sa is True:
            x_sa = self.sa1(x)
            x = torch.mul(x, x_sa)
        logits_1 = torch.sigmoid(self.outc1(x))
        x = self.up2(x, x3)
        if self.with_sa is True:
            x_sa = self.sa2(x)
            x = torch.mul(x, x_sa)
        logits_2 = torch.sigmoid(self.outc2(x))
        x = self.up3(x, x2)
        if self.with_sa is True:
            x_sa = self.sa3(x)
            x = torch.mul(x, x_sa)
        logits_3 = torch.sigmoid(self.outc3(x))
        x = self.up4(x, x1)
        if self.with_sa is True:
            x_sa = self.sa4(x)
            x = torch.mul(x, x_sa)
        logits_4 = torch.sigmoid(self.outc4(x))
        if self.training:
            #return logits_4
            return [logits_4, logits_3, logits_2, logits_1, logits_0]
        else:
            return logits_4

class UNet_CS(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, with_sa=False, model_factor=1):
        super(UNet_CS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.with_sa = with_sa
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channel_list[3], channel_list[4] // factor)
        if self.with_ca is True:
            self.ca0 = ChannelwiseAttention(channel_list[2])
            self.ca1 = ChannelwiseAttention(channel_list[3])
            self.ca2 = ChannelwiseAttention(channel_list[4] // factor)
            self.ca3 = ChannelwiseAttention(channel_list[3])
            self.ca4 = ChannelwiseAttention(channel_list[2])
        if self.with_sa is True:
            self.sa0 = SpatialAttention(channel_list[0])
            self.sa1 = SpatialAttention(channel_list[1])
            self.sa2 = SpatialAttention(channel_list[1])
            self.sa3 = SpatialAttention(channel_list[0])
        self.outc0= OutConv(channel_list[4] // factor, self.n_classes)
        self.up1 = Up(channel_list[4], channel_list[3] // factor, bilinear)
        self.outc1= OutConv(channel_list[3] // factor, self.n_classes)
        self.up2 = Up(channel_list[3], channel_list[2] // factor, bilinear)
        self.outc2 = OutConv(channel_list[2] // factor, self.n_classes)
        self.up3 = Up(channel_list[2], channel_list[1] // factor, bilinear)
        self.outc3 = OutConv(channel_list[1] // factor, self.n_classes)
        self.up4 = Up(channel_list[1], channel_list[0], bilinear)
        self.outc4 = OutConv(channel_list[0] // factor, self.n_classes)

        init_modules(self)

    def forward(self, x):
        x1 = self.inc(x)
        if self.with_sa is True:
            x_sa = self.sa0(x1)
            x1 = torch.mul(x1, x_sa)
        x2 = self.down1(x1)
        if self.with_sa is True:
            x_sa = self.sa1(x2)
            x2 = torch.mul(x2, x_sa)
        x3 = self.down2(x2)
        if self.with_ca is True:
            x_ca, _ = self.ca0(x3)
            x3 = torch.mul(x3, x_ca)
        x4 = self.down3(x3)
        if self.with_ca is True:
            x_ca, _ = self.ca1(x4)
            x4 = torch.mul(x4, x_ca)
        x5 = self.down4(x4)
        if self.with_ca is True:
            x5_ca, _ = self.ca2(x5)
            x5 = torch.mul(x5, x5_ca)
        logits_0 = torch.sigmoid(self.outc0(x5))
        x = self.up1(x5, x4)
        if self.with_ca is True:
            x_ca, _ = self.ca3(x)
            x = torch.mul(x, x_ca)
        logits_1 = torch.sigmoid(self.outc1(x))
        x = self.up2(x, x3)
        if self.with_ca is True:
            x_ca, _ = self.ca4(x)
            x = torch.mul(x, x_ca)
        logits_2 = torch.sigmoid(self.outc2(x))
        x = self.up3(x, x2)
        if self.with_sa is True:
            x_sa = self.sa2(x)
            x = torch.mul(x, x_sa)
        logits_3 = torch.sigmoid(self.outc3(x))
        x = self.up4(x, x1)
        if self.with_sa is True:
            x_sa = self.sa3(x)
            x = torch.mul(x, x_sa)
        logits_4 = torch.sigmoid(self.outc4(x))

        if self.training:
            #return logits_4
            return [logits_4, logits_3, logits_2, logits_1, logits_0]
        else:
            return logits_4
'''
class UNet_encoder_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, with_sa=False, model_factor=1):
        super(UNet_encoder_decoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.with_sa = with_sa
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channel_list[3], channel_list[4] // factor)
        if self.with_ca is True:
            self.ca0 = ChannelwiseAttention(channel_list[0])
            self.ca1 = ChannelwiseAttention(channel_list[1])
            self.ca2 = ChannelwiseAttention(channel_list[2])
            self.ca3 = ChannelwiseAttention(channel_list[3])
            self.ca4 = ChannelwiseAttention(channel_list[4] // factor)
        if self.with_sa is True:
            #self.sa0 = SpatialAttention(channel_list[4] // factor)
            self.conv_1 = CorrConv(channel_list[4] //factor, channel_list[3])
            self.sa1 = SpatialAttention(channel_list[3])
            self.conv_2 = CorrConv(channel_list[3], channel_list[2])
            self.sa2 = SpatialAttention(channel_list[2])
            self.conv_3 = CorrConv(channel_list[2], channel_list[1])
            self.sa3 = SpatialAttention(channel_list[1])
            self.conv_4 = CorrConv(channel_list[1], channel_list[0])
            self.sa4 = SpatialAttention(channel_list[0])
        self.outc0= OutConv(channel_list[4] // factor, self.n_classes)
        self.up1 = Up(channel_list[4], channel_list[3], bilinear)
        self.outc1= OutConv(channel_list[3], self.n_classes)
        self.up2 = Up(channel_list[3], channel_list[2], bilinear)
        self.outc2 = OutConv(channel_list[2], self.n_classes)
        self.up3 = Up(channel_list[2], channel_list[1], bilinear)
        self.outc3 = OutConv(channel_list[1], self.n_classes)
        self.up4 = Up(channel_list[1], channel_list[0], bilinear)
        self.outc4 = OutConv(channel_list[0], self.n_classes)

        init_modules(self)

    def forward(self, x):
        print(f'in conv input size: {x.shape}' )
        x1 = self.inc(x)
        if self.with_ca is True:
            x_ca, _ = self.ca0(x1)
            x1 = torch.mul(x1, x_ca)
        print(f'in conv output size: {x1.shape}' )
        print(f'E block 1 input size: {x1.shape}' )
        x2 = self.down1(x1)
        if self.with_ca is True:
            x_ca, _ = self.ca1(x2)
            x2 = torch.mul(x2, x_ca)
        print(f'E block 1 output size: {x2.shape}' )
        print(f'E block 2 input size: {x2.shape}' )
        x3 = self.down2(x2)
        if self.with_ca is True:
            x_ca, _ = self.ca2(x3)
            x3 = torch.mul(x3, x_ca)
        print(f'E block 2 output size: {x3.shape}' )
        print(f'E block 3 input size: {x3.shape}' )
        x4 = self.down3(x3)
        if self.with_ca is True:
            x_ca, _ = self.ca3(x4)
            x4 = torch.mul(x4, x_ca)
        print(f'E block 3 output size: {x4.shape}' )
        print(f'E block 4 input size: {x4.shape}' )
        x5 = self.down4(x4)
        if self.with_ca is True:
            x5_ca, _ = self.ca4(x5)
            x5 = torch.mul(x5, x5_ca)
        print(f'E block 4 output size: {x5.shape}' )
        print(f'D block 1 input size: {x5.shape}' )
        #logits_0 = torch.sigmoid(self.outc0(x5))
        x4 = self.up1(x5, x4)
        if self.with_sa is True:
            x5 = self.conv_1(x5)
            x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
            diffY = x4.size()[2] - x5.size()[2]
            diffX = x4.size()[3] - x5.size()[3]

            x5 = F.pad(x5, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x_sa = self.sa1(x5)
            x4 = torch.mul(x4, x_sa)
        print(f'D block 1 output size: {x4.shape}' )
        print(f'D block 2 input size: {x4.shape}' )
        logits_1 = torch.sigmoid(self.outc1(x4))
        x3 = self.up2(x4, x3)
        if self.with_sa is True:
            x4 = self.conv_2(x4)
            x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
            diffY = x3.size()[2] - x4.size()[2]
            diffX = x3.size()[3] - x4.size()[3]

            x4 = F.pad(x4, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x_sa = self.sa2(x4)
            x3 = torch.mul(x3, x_sa)
        print(f'D block 2 output size: {x3.shape}' )
        print(f'D block 3 input size: {x3.shape}' )
        logits_2 = torch.sigmoid(self.outc2(x3))
        x2 = self.up3(x3, x2)
        if self.with_sa is True:
            x3 = self.conv_3(x3)
            x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
            diffY = x2.size()[2] - x3.size()[2]
            diffX = x2.size()[3] - x3.size()[3]

            x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x_sa = self.sa3(x3)
            x2 = torch.mul(x2, x_sa)
        print(f'D block 3 output size: {x2.shape}' )
        print(f'D block 4 input size: {x2.shape}' )
        logits_3 = torch.sigmoid(self.outc3(x2))
        x1 = self.up4(x2, x1)
        if self.with_sa is True:
            x2 = self.conv_4(x2)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]

            x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x_sa = self.sa4(x2)
            x1 = torch.mul(x1, x_sa)
        print(f'D block 4 output size: {x1.shape}' )
        logits_4 = torch.sigmoid(self.outc4(x1))
        print(f'out conv output size: {logits_4.shape}' )

        if self.training:
            #return logits_4
            return [logits_4, logits_3, logits_2, logits_1]
        else:
            return logits_4
'''

class UNet_encoder_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, with_sa=False, model_factor=1):
        super(UNet_encoder_decoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.with_sa = with_sa
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        '''
        four encoder blocks
        '''
        self.eb1 = Encoder_block(channel_list[0], channel_list[1], with_ca=self.with_ca)
        self.eb2 = Encoder_block(channel_list[1], channel_list[2], with_ca=self.with_ca)
        self.eb3 = Encoder_block(channel_list[2], channel_list[3], with_ca=self.with_ca)
        factor = 2 if bilinear else 1
        self.eb4 = Encoder_block(channel_list[3], channel_list[4] // factor, with_ca=self.with_ca)
        '''
        Four decoder blocks
        '''
        self.db1 = Decoder_block(channel_list[4] // factor, channel_list[3], n_classes=self.n_classes, bilinear=self.bilinear, with_sa=self.with_sa)
        self.db2 = Decoder_block(channel_list[3], channel_list[2], n_classes=self.n_classes, bilinear=self.bilinear, with_sa=self.with_sa)
        self.db3 = Decoder_block(channel_list[2], channel_list[1], n_classes=self.n_classes, bilinear=self.bilinear, with_sa=self.with_sa)
        self.db4 = Decoder_block(channel_list[1], channel_list[0], n_classes=self.n_classes, bilinear=self.bilinear, with_sa=self.with_sa)

        init_modules(self)

    def forward(self, x):
        '''
        print(f'in conv input size: {x.shape}' )
        x1 = self.inc(x)
        print(f'in conv output size: {x1.shape}' )
        print(f'E block 1 input size: {x1.shape}' )
        x2 = self.eb1(x1)
        print(f'E block 1 output size: {x2.shape}' )
        print(f'E block 2 input size: {x2.shape}' )
        x3 = self.eb2(x2)
        print(f'E block 2 output size: {x3.shape}' )
        print(f'E block 3 input size: {x3.shape}' )
        x4 = self.eb3(x3)
        print(f'E block 3 output size: {x4.shape}' )
        print(f'E block 4 input size: {x4.shape}' )
        x5 = self.eb4(x4)
        print(f'E block 4 output size: {x5.shape}' )
        print(f'D block 1 input size: {x5.shape}' )
        #logits_0 = torch.sigmoid(self.outc0(x5))
        x4, logits_1 = self.db1(x5, x4)
        print(f'D block 1 output size: {x4.shape}' )
        print(f'D block 2 input size: {x4.shape}' )
        x3, logits_2 = self.db2(x4, x3)
        print(f'D block 2 output size: {x3.shape}' )
        print(f'D block 3 input size: {x3.shape}' )
        x2, logits_3 = self.db3(x3, x2)
        print(f'D block 3 output size: {x2.shape}' )
        print(f'D block 4 input size: {x2.shape}' )
        x1, logits_4 = self.db4(x2, x1)
        print(f'D block 4 output size: {x1.shape}' )
        print(f'out conv output size: {logits_4.shape}' )
        '''
        x1 = self.inc(x)
        x2 = self.eb1(x1)
        x3 = self.eb2(x2)
        x4 = self.eb3(x3)
        x5 = self.eb4(x4)
        x4, logits_1 = self.db1(x5, x4)
        x3, logits_2 = self.db2(x4, x3)
        x2, logits_3 = self.db3(x3, x2)
        x1, logits_4 = self.db4(x2, x1)
        if self.training:
            return [logits_4, logits_3, logits_2, logits_1]
        else:
            return logits_4

class UNet_multi(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, model_factor=1):
        super(UNet_multi, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channel_list[3], channel_list[4] // factor)
        if self.with_ca is True:
            self.ca = ChannelwiseAttention(channel_list[4] // factor)
        self.up1 = Up(channel_list[4], channel_list[3] // factor, bilinear)
        self.up2 = Up(channel_list[3], channel_list[2] // factor, bilinear)
        self.up3 = Up(channel_list[2], channel_list[1] // factor, bilinear)
        self.up4 = Up(channel_list[1], channel_list[0], bilinear)
        self.outc = OutConv(channel_list[0], n_classes)
        init_modules(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.with_ca is True:
            x5_ca, _ = self.ca(x5)
            x5 = torch.mul(x5, x5_ca)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_3d(nn.Module):
    def __init__(self, n_channels, n_classes, n_frame, bilinear=False, with_time_module=False):
        super(UNet_3d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_frame = n_frame
        self.with_time_module = with_time_module

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_1 = OutConv(64, n_classes)
        if with_time_module is True:
            self.corr_s_0 = CorrConv(64, 64)
            self.corr_m_0 = CorrConv(64 * (self.n_frame - 1), 64)
            self.corr_1 = CorrConv(64 * 2, 64)
            self.out_corr = OutConv(64, n_channels)

        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) #(B * n, C, H, W)  (B * n, 64, H, W)
        x_m = einops.rearrange(x, '(b n) c h w -> b n c h w', n = self.n_frame) #(B, n, c, h, w)
        x_0 = x_m[:, 0, :]
        logits = torch.sigmoid(self.outc_1(x_0))
        logits_corr = list()
        if self.with_time_module is True:
            for i in range(self.n_frame):
                sl_key = x_m[:, i, :]
                sl_1 = x_m[:, 0:i, :]
                sl_2 = x_m[:, i+1:self.n_frame,:]
                sl_corr = torch.cat([sl_1, sl_2], 1)
                sl_corr = einops.rearrange(sl_corr, 'b n c h w -> b (n c) h w')
                sl_key = self.corr_s_0(sl_key)
                sl_corr = self.corr_m_0(sl_corr)
                sl_merge = torch.cat([sl_key, sl_corr], 1)
                sl_merge = self.corr_1(sl_merge)
                sl_merge = torch.sigmoid(self.out_corr(sl_merge))
                logits_corr.append(sl_merge)
        return logits, logits_corr

class UNet_3d_with_CA(nn.Module):
    def __init__(self, n_channels, n_classes, n_frame, bilinear=False, with_time_module=False):
        super(UNet_3d_with_CA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_frame = n_frame
        self.with_time_module = with_time_module

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.ca = ChannelwiseAttention(1024)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_1 = OutConv(64, n_classes)
        if with_time_module is True:
            self.corr_s_0 = CorrConv(64, 64)
            self.corr_m_0 = CorrConv(64 * (self.n_frame - 1), 64)
            self.corr_1 = CorrConv(64 * 2, 64)
            self.corr_2 = CorrConv(64, n_classes)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_ca, _ = self.ca(x5)
        x5 = torch.mul(x5, x5_ca)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) #(B * n, C, H, W)  (B * n, 64, H, W)
        x_m = einops.rearrange(x, '(b n) c h w -> b n c h w', n = self.n_frame) #(B, n, c, h, w)
        x_0 = x_m[:, 0, :]
        logits = self.outc_1(x_0)
        logits_corr = list()
        if self.with_time_module is True:
            for i in range(self.n_frame):
                sl_key = x_m[:, i, :]
                sl_1 = x_m[:, 0:i, :]
                sl_2 = x_m[:, i+1:self.n_frame,:]
                sl_corr = torch.cat([sl_1, sl_2], 1)
                sl_corr = einops.rearrange(sl_corr, 'b n c h w -> b (n c) h w')
                sl_key = self.corr_s_0(sl_key)
                sl_corr = self.corr_m_0(sl_corr)
                sl_merge = torch.cat([sl_key, sl_corr], 1)
                sl_merge = self.corr_1(sl_merge)
                sl_merge = self.corr_2(sl_merge)
                logits_corr.append(sl_merge)
        return logits, logits_corr

class UNet_3d_with_SA(nn.Module):
    def __init__(self, n_channels, n_classes, n_frame, bilinear=False, with_time_module=False):
        super(UNet_3d_with_SA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_frame = n_frame
        self.with_time_module = with_time_module

        self.inc = DoubleConv(n_channels, 64)
        self.sa = SpatialAttention(64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_1 = OutConv(64, n_classes)
        if with_time_module is True:
            self.corr_s_0 = CorrConv(64, 64)
            self.corr_m_0 = CorrConv(64 * (self.n_frame - 1), 64)
            self.corr_1 = CorrConv(64 * 2, 64)
            self.corr_2 = CorrConv(64, n_classes)
        

    def forward(self, x):
        x1 = self.inc(x)
        x1_sa = self.sa(x1)
        x1 = torch.mul(x1, x1_sa)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) #(B * n, C, H, W)  (B * n, 64, H, W)
        x_m = einops.rearrange(x, '(b n) c h w -> b n c h w', n = self.n_frame) #(B, n, c, h, w)
        x_0 = x_m[:, 0, :]
        logits = self.outc_1(x_0)
        logits_corr = list()
        if self.with_time_module is True:
            for i in range(self.n_frame):
                sl_key = x_m[:, i, :]
                sl_1 = x_m[:, 0:i, :]
                sl_2 = x_m[:, i+1:self.n_frame,:]
                sl_corr = torch.cat([sl_1, sl_2], 1)
                sl_corr = einops.rearrange(sl_corr, 'b n c h w -> b (n c) h w')
                sl_key = self.corr_s_0(sl_key)
                sl_corr = self.corr_m_0(sl_corr)
                sl_merge = torch.cat([sl_key, sl_corr], 1)
                sl_merge = self.corr_1(sl_merge)
                sl_merge = self.corr_2(sl_merge)
                logits_corr.append(sl_merge)
        return logits, logits_corr

class MS_depth_net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_ca=False, model_factor=1):
        super(MS_depth_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_ca = with_ca
        self.model_factor = model_factor
        channel_list = [64, 128, 256, 512, 1024]
        channel_list = [int(item * self.model_factor) for item in channel_list]

        
        self.inc = DoubleConv(n_channels, channel_list[0])
        '''
        Decoder for MS
        '''
        self.down1_ms = Down(channel_list[0], channel_list[1])
        self.down2_ms = Down(channel_list[1], channel_list[2])
        self.down3_ms = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4_ms = Down(channel_list[3], channel_list[4] // factor)
        if self.with_ca is True:
            self.ca = ChannelwiseAttention(channel_list[4] // factor)
        
        '''
        Encoder for depth
        '''
        self.down1_dep = Down(channel_list[0], channel_list[1])
        self.down2_dep = Down(channel_list[1], channel_list[2])
        self.down3_dep = Down(channel_list[2], channel_list[3])
        factor = 2 if bilinear else 1
        self.down4_dep = Down(channel_list[3], channel_list[4] // factor)

        '''Decoder for depth'''
        self.up1_dep = Up(channel_list[4], channel_list[3] // factor, bilinear)
        self.up2_dep = Up(channel_list[3], channel_list[2] // factor, bilinear)
        self.up3_dep = Up(channel_list[2], channel_list[1] // factor, bilinear)
        self.up4_dep = Up(channel_list[1], channel_list[0], bilinear)
        self.outc_dep = OutConv(channel_list[0], 1)

        self.conv_c = CorrConv(channel_list[4] * 2, channel_list[4])
        '''
        Decoder for MS
        '''
        self.up1_ms = Up(channel_list[4], channel_list[3] // factor, bilinear)
        self.up2_ms = Up(channel_list[3], channel_list[2] // factor, bilinear)
        self.up3_ms = Up(channel_list[2], channel_list[1] // factor, bilinear)
        self.up4_ms = Up(channel_list[1], channel_list[0], bilinear)
        self.outc_ms = OutConv(channel_list[0], n_classes)

        init_modules(self)

    def forward(self, x):
        x1 = self.inc(x)

        '''encoder layer for ms'''
        x2_ms = self.down1_ms(x1)
        x3_ms = self.down2_ms(x2_ms)
        x4_ms = self.down3_ms(x3_ms)
        x5_ms = self.down4_ms(x4_ms)
        if self.with_ca is True:
            x5_ca, _ = self.ca(x5_ms)
            x5_ms = torch.mul(x5_ms, x5_ca)
        '''encoder layer for depth'''
        x2_dep = self.down1_ms(x1)
        x3_dep = self.down2_ms(x2_dep)
        x4_dep = self.down3_ms(x3_dep)
        x5_dep = self.down4_ms(x4_dep)

        '''decoder layer for depth'''
        x_dep = self.up1_dep(x5_dep, x4_dep)
        x_dep = self.up2_dep(x_dep,  x3_dep)
        x_dep = self.up3_dep(x_dep,  x2_dep)
        x_dep = self.up4_dep(x_dep,  x1)

        out_dep = post_process_depth(self.outc_dep(x_dep))
        
        '''concat the ms feature map and depth feature map'''
        x5_concat = torch.cat((x5_ms, x5_dep), dim=1)
        x5 = self.conv_c(x5_concat)

        '''decoder layer for MS'''
        x = self.up1_ms(x5,x4_ms)
        x = self.up2_ms(x, x3_ms)
        x = self.up3_ms(x, x2_ms)
        x = self.up4_ms(x, x1)
        out_ms = self.outc_ms(x)

        return out_ms, out_dep


if __name__ == '__main__':
    input = torch.randn(4, 3, 260, 346)
    net = UNet_3d_with_SA(3, 2)
    output = net(input)
    