import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

def init_modules(net):
    print("init module !")
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            xavier_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)