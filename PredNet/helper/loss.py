#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Loss module (Choosing the correct loss generally)
"""

import torch
import torch.nn.functional as f
from pytorch_msssim import ssim

def loss(x, y, loss):
    """
    Only used for testing / validation
    
    x := predicted next frame
    y := true next frame
    loss := name of the loss <mse|mae|bce|bcel|ssim>
    """
    if loss == 'mse':
        return f.mse_loss(x, y)
    elif loss == 'mae':
        return f.l1_loss(x, y)
    elif loss == 'bce':
        return f.binary_cross_entropy(x, y)
    elif loss == 'bcel':
        return f.binary_cross_entropy_with_logits(x, y)
    elif loss == 'ssim':
        # The library is only for 4d tensors, but we have 5d (TxBxCxHxW)
        # Loop through the sequence and take mean ssim
        loss = 0
        for i in range(len(x)):
            # ssim is not a loss function -> 1 - ssim
            loss += 1 - ssim(x[i], y[i], data_range=torch.max(y[i]))
        return loss / len(x)
    else:
        raise IOError('[ERROR] Use a valid loss function <mse|mae|bce|bcel|ssim>')


def error_loss(time_weight, layer_weight, seq_len, error):
    """
    Compute the error for PredNet module / Used in training
    
    time_weight := list of specified weights for the sequence length
    layer_weight := list of specified weights for the layer length
    seq_len := length of input sequence (MUST match len(time_weight))
    error := the error values (batch_size x layer x time)
    """
    bsize = error.size(0)
    # (bsize * layer,time) x (time,1) = (bsize * layer,1)
    error = torch.mm(torch.reshape(error, (-1, seq_len)), time_weight)
    # (bsize, layer) x (layer, 1) = (bsize, 1)
    error = torch.mm(torch.reshape(error, (bsize, -1)), layer_weight)
    error = torch.mean(error)

    return error
