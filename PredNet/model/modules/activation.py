#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 16.06.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Some additional activation functions
"""  

import torch


def hardSig(x):
    """
    Hard sigmoid function
    
    x := input parameter
    """
    return torch.max(torch.min(0.25 * x + 0.5,
                               torch.ones(x.size()).cuda()),
                               torch.zeros(x.size()).cuda())
