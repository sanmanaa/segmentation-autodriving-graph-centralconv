# Most of the code below is from the following repo:
#  https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
#
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from network.mynn import initialize_weights, Norm2d
from runx.logx import logx
from config import cfg
import math

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import FloatTensor
from torch.nn.modules.module import Module
import torch


from einops import rearrange

BN_MOMENTUM = 0.1
align_corners = cfg.MODEL.ALIGN_CORNERS
relu_inplace = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = Norm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)


    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logx.msg(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logx.msg(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logx.msg(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != (num_channels[branch_index] *
                                                 block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(num_channels[branch_index] * block.expansion,
                       momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
 
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        Norm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                Norm2d(num_outchannels_conv3x3,
                                       momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                Norm2d(num_outchannels_conv3x3,
                                       momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
   
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))    
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        extra = cfg.MODEL.OCR_EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        self.conv1 = InceptionA(in_channels=3)
        self.bn1 = Norm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = Norm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        print(block)
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.high_level_ch = int(np.sum(pre_stage_channels))
        self.RFP=RFP()
        self.gnn1=MGL(channel=192)
        # self.gnn2=MGL(channel=384)
        # self.hanet0 = HANet_Conv(48, 96)
        # initialize_weights(self.hanet0)
        # self.hanet1 = HANet_Conv(96, 192)
        # initialize_weights(self.hanet1)
        # self.hanet2 = HANet_Conv(192,384)
        # initialize_weights(self.hanet2)




    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        Norm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        # nn.Conv2d(
                        #     inchannels, outchannels, 3, 2, 1, bias=False),
                        InceptionB(inchannels,
                                  outchannels),
                        Norm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x_in,pos=None):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)


        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        x_list[2]=self.gnn1(x_list[2])

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # x_list[3]=self.gnn2(x_list[3])
        #x_list[3]=self.hanet2(x_list[2],x_list[3],pos)#changed
        x = self.stage4(x_list)

        feats=self.RFP(x[0],x[1],x[2],x[3])

        #print(x[2].shape)#[192,32,50]/[64,100]
        #print(x[3].shape)#([2, 384, 16, 25])/([2, 384, 32, 50])

        # Upsampling
        # x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(x[1], size=(x0_h, x0_w),
        #                    mode='bilinear', align_corners=align_corners)
        # x2 = F.interpolate(x[2], size=(x0_h, x0_w),
        #                    mode='bilinear', align_corners=align_corners)
        # x3 = F.interpolate(x[3], size=(x0_h, x0_w),
        #                    mode='bilinear', align_corners=align_corners)

        # feats = torch.cat([x[0], x1, x2, x3], 1)
        #([2, 720, 128, 200]),([2, 720, 256, 400])

        return None, None, feats

    def init_weights(self, pretrained=cfg.MODEL.HRNET_CHECKPOINT):
        pretrained=False#changed
        logx.msg('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                #print('skipped', name)
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, cfg.MODEL.BNFUNC):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,
                                         map_location={'cuda:0': 'cpu'})
            logx.msg('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer',
                                         'aux_head').replace('model.', ''): v
                               for k, v in pretrained_dict.items()}
            #print(set(model_dict) - set(pretrained_dict))
            #print(set(pretrained_dict) - set(model_dict))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))


def get_seg_model():
    model = HighResolutionNet()
    model.init_weights()

    return model


class Conv2d_cd(nn.Module): # 290
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, groups=1, theta=0.4):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #if out_normal.shape[3]!=out_diff.shape[3]:
            # print(out_normal.shape)
            # print(out_diff.shape)
            return out_normal - self.theta * out_diff
    #self.proj = Conv2d_cd(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, theta=0.6) #329

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.conv11=Conv2d_cd(in_channels, 21, kernel_size=3, stride=2, padding=1,theta=0.4)
        self.conv12=Conv2d_cd(in_channels, 21, kernel_size=5, stride=2, padding=2,theta=0.4)
        self.conv13=Conv2d_cd(in_channels, 22, kernel_size=7, stride=2, padding=3,theta=0.4)
    def forward(self,x):
        conv11=self.conv11(x)
        conv12=self.conv12(x)
        conv13=self.conv13(x)
        outputs=[conv11,conv12,conv13]
        return torch.cat(outputs,dim=1)
    
class InceptionB(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionB,self).__init__()
        out_channels=int(out_channels/3)
        self.conv11=Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=2, padding=1,theta=0.4)
        self.conv12=Conv2d_cd(in_channels, out_channels, kernel_size=5, stride=2, padding=2,theta=0.4)
        self.conv13=Conv2d_cd(in_channels, out_channels, kernel_size=7, stride=2, padding=3,theta=0.4)
    def forward(self,x):
        conv11=self.conv11(x)
        conv12=self.conv12(x)
        conv13=self.conv13(x)
        outputs=[conv11,conv12,conv13]
        return torch.cat(outputs,dim=1)

class MGL(nn.Module):
    # multi-graph layer
    def __init__(self, channel, dilation=1):
        super(MGL, self).__init__()
        self.fold = nn.Unfold(kernel_size=3, padding=dilation, dilation=dilation)
        self.conv = nn.ModuleList([BasicConv2d(channel, channel, 1) for _ in range(3)])

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, c, h, w)
        x1 = self.conv[0](x) #
        x2 = self.conv[1](x)
        x3 = self.conv[2](x)
        x1 = rearrange(self.fold(x1), '(n) (c k2) hw -> n hw k2 c', c=c) # n, hw, t, kk, c
        x1 = rearrange(x1, 'n hw k2 c -> n hw (k2) c') # n, hw, tkk, c
        x2 = rearrange(self.fold(x2), '(n) (c k2) hw -> n hw c k2', c=c) # n, hw, c, t, kk
        x2 = rearrange(x2, 'n hw c k2 -> n hw c k2') # n, hw, c, tkk

        score1 = F.softmax(torch.matmul(x1, x2), dim=-1) # n, hw, tkk, tkk

        x3 = rearrange(self.fold(x3), '(n) (c k2) hw -> n hw (k2 c)',  c=c) # n, hw, t, kkc
        x3 = rearrange(x3, 'n hw (k2 c) -> n hw (k2) c', c=c) # n, hw, tkk, c
        x5 = torch.matmul(score1, x3) # n, hw, tkk, c 

        kk = x5.shape[2]
        center = x5[:, :,  kk//2, :].unsqueeze(-1) # n, hw, t, c, 1
        score4 = F.softmax(torch.matmul(x5, center), dim=-2) # n, hw, t, kk, 1
        x = torch.sum(x5*score4, dim=-2) # n, hw, t, c
        x = rearrange(x, 'n (h w) c -> n c h w', h=h)
        return x
    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn=False):
        super(BasicConv2d, self).__init__()
        self.use_bn = bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out
    
class RFP(nn.Module):
    def __init__(self):
        super(RFP, self).__init__()
        self.conv1 = nn.Conv2d(48, 720, 1, 1, 0)
        self.conv2 = nn.Conv2d(96, 720, 1, 1, 0)
        self.conv3 = nn.Conv2d(192, 720, 1, 1, 0)
        self.conv4 = nn.Conv2d(384, 720, 1, 1, 0)

        self.fpn_convs = nn.Conv2d(720, 720, 3, 1, 1)
        self.rfp_aspp = ASPP(720, 720 // 4)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x0,x1,x2,x3):
        P4_ = self.conv4(x3)#[3,720,16,25]
        P3_ = self.conv3(x2)#[3,720,32,50]       
        P2_ = self.conv2(x1)#[3,720,64,100]
        P1_ = self.conv1(x0)#[3,720,128,200]
 
        size2 = P2_.shape[2:]
        size1 = P1_.shape[2:]
        size3 = P3_.shape[2:]
        
        P3 = P3_ + F.interpolate(P4_, size=size3, mode='nearest')
        P2 = P2_ + F.interpolate(P3_, size=size2, mode='nearest')
        P1 = P1_ + F.interpolate(P2_, size=size1, mode='nearest')

        P1 = self.fpn_convs(P1)#[3,256,208,208]
        P2 = self.fpn_convs(P2) #[3,256,208,208]  #fpn的输出
        P3 = self.fpn_convs(P3)
        P4 = self.fpn_convs(P4_)

        C1=self.rfp_aspp(P1)#[3,256,208,208]
        C2=self.rfp_aspp(P2)#[3,256,208,208]
        C3=self.rfp_aspp(P3)
        C4=self.rfp_aspp(P4)
        
        P3_1 = C3 + F.interpolate(C4, size=size3, mode='nearest')
        P2_1 = C2 + F.interpolate(P3_1, size=size2, mode='nearest')
        P1_1 = C1 + F.interpolate(P2_1, size=size1, mode='nearest')
 
        P1_1 = self.fpn_convs(P1_1)
        P2_1 = self.fpn_convs(P2_1)   #fpn的输出

        layer0=P1_1+C1
        layer0=self.relu(layer0)
        # print(layer0.shape)

        return layer0
