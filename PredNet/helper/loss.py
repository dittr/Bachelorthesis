#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Loss module (Choosing the correct loss generally)
""" 

import torch.nn.functional as f

def loss(x, y, loss):
    """
    """
    if loss == 'mse':
        return f.mse_loss(x, y)
    elif loss == 'mae':
        return f.l1_loss(x, y)
    elif loss == 'bce':
        return f.binary_cross_entropy(x, y)
    elif loss == 'bcel':
        return f.binary_cross_entropy_with_logits(x, y)
    else:
        raise IOError('[ERROR] Use a valid loss function <mse|mae|bce|bcel>')