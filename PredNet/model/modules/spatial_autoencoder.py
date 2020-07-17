#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 24.01.2020
@author: Soeren Dittrich
@version: 0.0.1
@description: Spatial autoencoder
""" 

import numpy as np
import torch
import torch.nn as nn

def pad_size(kernel):
    """
    """
    return int(np.floor(kernel / 2))


"""
Spatial Encoder
"""
class Encoder(nn.Module):
    """
    """
    def __init__(self, depth_in, depth_out, kernel_size, indices=False):
        super(Encoder, self).__init__()

        # Create padding size ones
        pad = pad_size(kernel=kernel_size[0])
        
        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out,
                              kernel_size=kernel_size[0], padding=pad,
                              bias=True)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size[1],
                                 return_indices=indices)
        
        self.init_weight_bias()
        
    """
    """
    def forward(self, x):       
        return self.pool(torch.tanh(self.conv(x)))
    
    """
    """
    def init_weight_bias(self):
        # Initialize the weights
        nn.init.xavier_normal_(self.conv.weight)
        
"""
Spatial Decoder with nearest-neighbor upsampling
"""
class Decoder(nn.Module):
    """
    """
    def __init__(self, depth_in, depth_out, kernel_size, scale):
        super(Decoder, self).__init__()
        
        # Create padding size ones
        pad = pad_size(kernel=kernel_size)
        self.scale = scale
        
        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out,
                              kernel_size=kernel_size, padding=pad,
                              bias=True)
        
        self.init_weight_bias()
        
    """
    """
    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=self.scale,
                         mode='nearest'))
        
    """
    """
    def init_weight_bias(self):
        # Initialize the weights
        nn.init.xavier_normal_(self.conv.weight)
        
"""
Spatial Decoder (3D) with nearest-neighbor upsampling
"""
class Decoder3d(nn.Module):
    """
    """
    def __init__(self, depth_in, depth_out, kernel_size, scale):
        super(Decoder3d, self).__init__()
        
        # Create padding size ones
        pad = pad_size(kernel=kernel_size)
        self.scale = scale
        
        self.conv = nn.Conv3d(in_channels=depth_in, out_channels=depth_out,
                              kernel_size=kernel_size, padding=pad,dilation=1,
                              stride=1, bias=True)
        
        self.init_weight_bias()
        
    """
    """
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        x = x[:,:,0,:,:][:,:,None,:,:]
        x = self.conv(x)
        
        return x
        
    """
    """
    def init_weight_bias(self):
        # Initialize the weights
        nn.init.xavier_normal_(self.conv.weight)
