"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from .submodules import *
from ._ms_util import init_modules
import einops

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        print(output.shape)
        output = self.fc(output)

        return output

class Recurrent_Down(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_block, stride, recurrent_block_type='convlstm'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.recurrent_block_type = recurrent_block_type
        self.stride = stride
        self.conv_x = self._make_layer(block, self.out_channels, num_block, self.stride)
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=self.out_channels,
                                               hidden_size=self.out_channels,
                                               kernel_size=3)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x, prev_state=None):
        x = self.conv_x(x)
        state = self.recurrent_block(x, prev_state)
        out = state[0] if self.recurrent_block_type == 'convlstm' else state
        return out, state


class ResNet_recurrent(nn.Module):

    def __init__(self, block, num_block, in_channels, n_classes, num_encoders=4, base_num_channels=32, 
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
        self.encoders.append(Recurrent_Down(block, 32, 64, num_block[0], 2, recurrent_block_type))
        self.encoders.append(Recurrent_Down(block, 64, 128, num_block[1], 2, recurrent_block_type))
        self.encoders.append(Recurrent_Down(block, 128, 256, num_block[2], 2, recurrent_block_type))
        self.encoders.append(Recurrent_Down(block, 256, 512, num_block[3], 2, recurrent_block_type))
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


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

def recurrent_resnet18(num_frame=4):
    """
    return a RNN with resnet18 backbone
    """
    return ResNet_recurrent(BasicBlock, [2, 2, 2, 2], 3, 1, num_frame=num_frame)

def recurrent_resnet34(num_frame=4):
    """
    return a RNN with resnet18 backbone
    """
    return ResNet_recurrent(BasicBlock, [3, 4, 6, 3], 3, 1, num_frame=num_frame)

if __name__ == '__main__':
    input = torch.ones((4, 3, 256, 336))
    model = resnet18()
    output = model(input)
    print(output.shape)