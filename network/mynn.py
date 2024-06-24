"""
Custom Norm wrappers to enable sync BN, regular BN and for weight
initialization
"""
import re
import torch
import torch.nn as nn
from config import cfg

#from apex import amp
from torch.cuda.amp import autocast
from runx.logx import logx


align_corners = cfg.MODEL.ALIGN_CORNERS


def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL, 'BNFUNC')
    normalization_layer = layer(in_channels, **kwargs)
    return normalization_layer


# def initialize_weights(*models):
#     """
#     Initialize Model Weights
#     """
#     for model in models:
#         for module in model.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     module.bias.data.zero_()
#             elif isinstance(module, cfg.MODEL.BNFUNC):
#                 module.weight.data.fill_(1)
#                 module.bias.data.zero_()

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


#@amp.float_function
@autocast()
def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=align_corners)


#@amp.float_function
@autocast()
def Upsample2(x):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                     align_corners=align_corners)


def Down2x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=0.5, mode='bilinear', align_corners=align_corners)


def Up15x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=1.5, mode='bilinear', align_corners=align_corners)


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    if cfg.OPTIONS.TORCH_VERSION >= 1.5:
        x_scaled = torch.nn.functional.interpolate(
            x, size=y_size, mode='bilinear',
            align_corners=align_corners)
    else:
        x_scaled = torch.nn.functional.interpolate(
            x, size=y_size, mode='bilinear',
            align_corners=align_corners)
    return x_scaled


def DownX(x, scale_factor):
    '''
    scale x to the same size as y
    '''
    if cfg.OPTIONS.TORCH_VERSION >= 1.5:
        x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners, recompute_scale_factor=True)
    else:
        x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners)
    return x_scaled


def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    '''
    if cfg.OPTIONS.TORCH_VERSION >= 1.5:
        x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners, recompute_scale_factor=True)
    else:
        x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners)
    return x_scaled


def initialize_embedding(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.zero_() #original



def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

def Zero_Masking(input_tensor, mask_org):
    output = input_tensor.clone()
    output.mul_(mask_org)
    return output

def RandomPosZero_Masking(input_tensor, p=0.5):
    output = input_tensor.clone()
    noise_b = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    noise_u = input_tensor.new().resize_(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
    noise_b.bernoulli_(1 - p)
    noise_b = noise_b.expand_as(input_tensor)
    output.mul_(noise_b)
    return output

def RandomVal_Masking(input_tensor, mask_org):
    output = input_tensor.clone()
    noise_u = input_tensor.new().resize_(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
    mask = (mask_org==0).type(input_tensor.type())
    mask = mask.expand_as(input_tensor)
    mask = torch.mul(mask, noise_u.uniform_(torch.min(input_tensor).item(), torch.max(input_tensor).item()))
    mask_org = mask_org.expand_as(input_tensor)
    output.mul_(mask_org)
    output.add_(mask)
    return output

def RandomPosVal_Masking(input_tensor, p=0.5):
    output = input_tensor.clone()
    noise_b = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    noise_u = input_tensor.new().resize_(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
    mask = noise_b.bernoulli_(1 - p)
    mask = (mask==0).type(input_tensor.type())
    mask = mask.expand_as(input_tensor)
    mask = torch.mul(mask, noise_u.uniform_(torch.min(input_tensor).item(), torch.max(input_tensor).item()))
    noise_b = noise_b.expand_as(input_tensor)
    output.mul_(noise_b)
    output.add_(mask)
    return output

def masking(input_tensor, p=0.5):
    output = input_tensor.clone()
    noise_b = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    noise_u = input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2), input_tensor.size(3))
    mask = noise_b.bernoulli_(1 - p)
    mask = (mask==0).type(input_tensor.type())
    mask.mul_(noise_u.uniform_(torch.min(input_tensor).item(), torch.max(input_tensor).item()))
    # mask.mul_(noise_u.uniform_(5, 10))
    noise_b = noise_b.expand_as(input_tensor)
    mask = mask.expand_as(input_tensor)
    output.mul_(noise_b)
    output.add_(mask)
    return output
