#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 26.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Error layer
"""  

import torch
import torch.nn as nn
import torch.nn.functional as f

class ErrorLayer(nn.Module):
    """
    """
    def __init__(self):
        """
        """
        super(ErrorLayer, self).__init__()
        
    def forward(self, x, y):
        """
        """
        return torch.cat((f.relu(x - y), f.relu(y - x)), 0)