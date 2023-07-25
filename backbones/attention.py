"""residual attention network in pytorch
[1] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
    Residual Attention Network for Image Classification
    https://arxiv.org/abs/1704.06904
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *
from ._ms_util import init_modules
import einops

#"""The Attention Module is built by pre-activation Residual Unit [11] with the
#number of channels in each stage is the same as ResNet [10]."""

class PreActResidualUnit(nn.Module):
    """PreAct Residual Unit
    Args:
        in_channels: residual unit input channel number
        out_channels: residual unit output channel numebr
        stride: stride of residual unit when stride = 2, downsample the featuremap
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channels = int(out_channels / 4)
        self.residual_function = nn.Sequential(
            #1x1 conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_channels, 1, stride),

            #3x3 conv
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),

            #1x1 conv
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, 1)
        )

        self.shortcut = nn.Sequential()
        if stride != 2 or (in_channels != out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):

        res = self.residual_function(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class AttentionModule1(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual
        #Units before splitting into trunk branch and mask branch. t denotes
        #the number of Residual Units in trunk branch. r denotes the number of
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown4 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup4 = self._make_residual(in_channels, out_channels, r)

        self.shortcut_short = PreActResidualUnit(in_channels, out_channels, 1)
        self.shortcut_long = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        ###We make the size of the smallest output map in each mask branch 7*7 to be consistent
        #with the smallest trunk output map size.
        ###Thus 3,2,1 max-pooling layers are used in mask branch with input size 56 * 56, 28 * 28, 14 * 14 respectively.
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 28
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #28 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut_long = self.shortcut_long(x_s)

        #seccond downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        #14 shortcut
        shape2 = (x_s.size(2), x_s.size(3))
        shortcut_short = self.soft_resdown3(x_s)

        #third downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown3(x_s)

        #mid
        x_s = self.soft_resdown4(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long

        #thrid upsample out 54
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual
        #Units before splitting into trunk branch and mask branch. t denotes
        #the number of Residual Units in trunk branch. r denotes the number of
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #14 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut = self.shortcut(x_s)

        #seccond downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        #mid
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule3(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #mid
        x_s = self.soft_resdown2(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class Attention(nn.Module):
    """residual attention netowrk
    Args:
        block_num: attention module number for each stage
    """

    def __init__(self, block_num, class_num=100):

        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(64, 256, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(256, 512, block_num[1], AttentionModule2)
        self.stage3 = self._make_stage(512, 1024, block_num[2], AttentionModule3)
        self.stage4 = nn.Sequential(
            PreActResidualUnit(1024, 2048, 2),
            PreActResidualUnit(2048, 2048, 1),
            PreActResidualUnit(2048, 2048, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.pre_conv(x)
        print(x.shape)
        x = self.stage1(x)
        print(x.shape)
        x = self.stage2(x)
        print(x.shape)
        x = self.stage3(x)
        print(x.shape)
        x = self.stage4(x)
        print(x.shape)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_stage(self, in_channels, out_channels, num, block):

        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 2))

        for _ in range(num):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

class Recurrent_Down(nn.Module):
    def __init__(self, block, in_channels, out_channels, num, recurrent_block_type='convlstm'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num = num
        self.block = block
        self.recurrent_block_type = recurrent_block_type
        self.conv_x = self._make_stage(self.in_channels, self.out_channels, self.num, self.block)
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=self.out_channels,
                                               hidden_size=self.out_channels,
                                               kernel_size=3)

    def _make_stage(self, in_channels, out_channels, num, block):

        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 2))

        for _ in range(num):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, prev_state=None):
        x = self.conv_x(x)
        state = self.recurrent_block(x, prev_state)
        out = state[0] if self.recurrent_block_type == 'convlstm' else state
        return out, state


class Attention_recurrent(nn.Module):

    def __init__(self, num_block, in_channels, n_classes, num_encoders=4, base_num_channels=32, 
                  num_frame=4, bilinear=False, recurrent_block_type='convlstm'):
        super().__init__()

        self.in_channels = in_channels
        self.num_frame = num_frame
        self.n_classes = n_classes
        self.num_block = num_block
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_encoder = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.encoders = nn.ModuleList()
        self.encoders.append(Recurrent_Down(AttentionModule1, 32, 64, num_block[0], recurrent_block_type))
        self.encoders.append(Recurrent_Down(AttentionModule2, 64, 128, num_block[1], recurrent_block_type))
        self.encoders.append(Recurrent_Down(AttentionModule3, 128, 256, num_block[2], recurrent_block_type))
        self.encoders.append(Recurrent_Down(AttentionModule3, 256, 512, num_block[3],recurrent_block_type))
        self.decoder = nn.ModuleList()
        self.decoder.append(Up(512, 256, bilinear))
        self.decoder.append(Up(256, 128, bilinear))
        self.decoder.append(Up(128, 64, bilinear))
        self.decoder.append(Up(64, 32, bilinear))
        self.outc = OutConv(32, self.n_classes)
        init_modules(self)


    def sub_forward(self, x, prev_states=None):
        x = self.conv1(x)
        if prev_states is None:
            prev_states = [None] * self.num_encoders
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            blocks.append(x)
            x, state = encoder(x, prev_states[i])
            states.append(state)
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, blocks[self.num_encoders-i-1])
        logits = torch.sigmoid(self.outc(x))
        return logits, states

    def forward(self, batch_images, prev_states=None):
        batch_images = einops.rearrange(batch_images, '(b n) c h w -> b n c h w', n = self.num_frame, c=self.in_channels) #(B, n, c, h, w)
        out_list = list()
        for i in range(self.num_frame):
            x = batch_images[:, i]  #b c h w
            x, prev_states = self.sub_forward(x, prev_states)  #b 1 h w
            out_list.append(x) 
        return torch.stack(out_list, dim=1)  #b n 1 h w
def attention56():
    return Attention([1, 1, 1])

def attention92():
    return Attention([1, 2, 3])

def attention_rnn_56(num_frame=4):
    return Attention_recurrent([1, 1, 1, 1], 3, 1, num_frame=num_frame)

if __name__ == '__main__':
    input = torch.ones((4, 3, 256, 336))
    model = attention_rnn_56()
    output = model(input)
    print(output.shape)