#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 26.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Input layer
"""

import torch.nn as nn
import torch.nn.functional as f

class InputLayer(nn.Module):
    """
    """
    def __init__(self, channel_in, channel_out, kernel_size, padding, stride):
        """
        """
        super(InputLayer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_out,
                              kernel_size=kernel_size,
                              padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        """
        """
        return self.pool(f.relu(self.conv(x)))
        