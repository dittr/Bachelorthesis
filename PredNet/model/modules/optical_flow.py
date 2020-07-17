#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 27.02.2020
@author: Soeren Dittrich
@version: 0.0.2
@description: Optical flow network
"""

import numpy as np
import torch
import torch.nn as nn

def pad_size(kernel):
    """
    """
    return int(np.floor(kernel / 2))

"""
"""
class OpticalFlow(nn.Module):
    """
    """
    def __init__(self, channel_in, channel_out, kernel_size,
                 clamp_min, clamp_max, size):
        super(OpticalFlow, self).__init__()
        
        self.min = clamp_min
        self.max = clamp_max
        
        self.height = size[0]
        self.width = size[1]
        
        # Create padding size ones
        padding = pad_size(kernel=kernel_size[0])
        
        # 15 x 15 layers
        self.conv1 = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_out,
                               kernel_size=kernel_size[0],
                               padding=padding, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel_out,
                               out_channels=channel_out,
                               kernel_size=kernel_size[0],
                               padding=padding, bias=True)
        
        # 1x1 layer
        self.conv3 = nn.Conv2d(in_channels=channel_out,
                               out_channels=channel_out,
                               kernel_size=kernel_size[1],
                               padding=0, bias=True)
        
        self.init_weight_bias()
        
    """
    """
    def init_weight_bias(self):
        # Initialize the weights
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        self.conv3.weight.data.fill_(0)
        
        # Initialize the biases
        self.conv3.bias.data.fill_(0)
        
    """
    """
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        
        x = x.permute(1,0,2,3)
        x1 = x[0]
        x2 = x[1]
        
        x = torch.stack((x1 * (2 / self.height),
                        x2 * (2 / self.width))).permute(1,0,2,3)
        
        return torch.clamp(x, self.min, self.max)
