from __future__ import division
import torch
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

def adaptative_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)

def predict_depth(in_planes, with_confidence):
    return nn.Conv2d(in_planes, 2 if with_confidence else 1, kernel_size=3, stride=1, padding=1, bias=True)


def post_process_depth(depth, activation_function=None, clamp=False):
    if activation_function is not None:
        depth = activation_function(depth)

    if clamp:
        depth = depth.clamp(10, 60)

    return depth[:,0]