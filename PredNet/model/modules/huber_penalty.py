#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 03.03.2020
@author: Soeren Dittrich
@version: 0.0.3
@description: Huber penalty
"""

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as auto

def pad_size(kernel):
    """
    """
    return int(np.floor(kernel / 2))

"""
"""
class HuberPenalty(nn.Module):
    """
    """
    def __init__(self, kernel_size, mu, l1, gpu=False, training=False):
        super(HuberPenalty, self).__init__()
        
        # Used as 5 point-stencil non-trainable weights
        if gpu:
            self.gradx = torch.zeros(3, 3).cuda()
            self.grady = torch.zeros(3, 3).cuda()
        else:
            self.gradx = torch.zeros(3, 3)
            self.grady = torch.zeros(3, 3)
        
        self.init_stencil()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size,
                               padding=1)

        self.init_weights_bias()

        self.huber = Huber()
        self.mu = mu
        self.l1 = l1
        self.training = training
        
    """
    """
    def set_training(self, training):
        self.training = training
        
    """
    based on the paper: -1/2 / 1/2 for both
    """
    def init_stencil(self, v1=-1/2, v2=1/2):
        # x dimension
        self.gradx[1][0] = v1
        self.gradx[1][2] = v2
        
        # y dimension
        self.grady[0][1] = v1
        self.grady[2][1] = v2
        
    """
    """
    def init_weights_bias(self, bias=0):
        # Initialize the weights
        self.conv1.weight.data[0][0] = self.grady
        self.conv2.weight.data[0][0] = self.gradx
        
        # Initialize the bias
        self.conv1.bias.data.fill_(bias)
        self.conv2.bias.data.fill_(bias)
        
    """
    """        
    def forward(self, x):
        grad = x.clone()

        grad = grad.permute(1,0,2,3)
        x1 = grad[0][None,:,:,:].permute(1,0,2,3)
        x2 = grad[1][None,:,:,:].permute(1,0,2,3)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        grad = torch.cat((x1, x2), 1)

        return self.huber.apply(x, grad, self.mu, self.l1)


"""
Return the input unchanged and only keep the input which was forwarded through
the "Huber-net". Then only return gradients through the net.
"""
class Huber(auto.Function):
    """
    """
    @staticmethod
    def forward(ctx, x, grad, mu, l1):
        ctx.save_for_backward(grad)
        ctx.mu = mu
        ctx.l1 = l1
        return x
        
    """
    """
    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        mu = ctx.mu
        l1 = ctx.l1
        
        grad[grad > mu] = mu
        grad[grad < -mu] = -mu
        grad *= l1 / grad.numel()
        
        return grad_input, grad, None, None
