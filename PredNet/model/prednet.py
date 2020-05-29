#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.3
@description: PredNet module
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

from model.modules.input import InputLayer
from model.modules.prediction import PredictionLayer
from model.modules.error import ErrorLayer
from model.modules.conv_lstm import CONV_LSTM as LstmLayer

class PredNet(nn.Module):
    """
    """
    def __init__(self, channels, kernel, padding, stride,
                 dropout, peephole, pixel_max, gpu=False):
        """
        """
        super(PredNet, self).__init__()
        self.name = 'prednet'
        self.channels = channels
        self.layer = len(self.channels) 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dropout = dropout
        self.peephole = peephole
        self.pixel_max = pixel_max
        self.gpu = gpu
        self.input = nn.ModuleList()
        self.prediction = nn.ModuleList()
        self.error = nn.ModuleList()
        self.lstm = nn.ModuleList()
        
        self.channels.append(0)
        
        for i in range(self.layer - 1):
            self.input.append(InputLayer(channel_in=self.channels[i] * 2,
                                         channel_out=self.channels[i+1],
                                         kernel_size=self.kernel,
                                         padding=self.padding,
                                         stride=self.stride))
        
        for i in range(self.layer):
            self.prediction.append(PredictionLayer(channel_in=self.channels[i],
                                                   channel_out=self.channels[i],
                                                   kernel_size=self.kernel,
                                                   padding=self.padding))
            self.error.append(ErrorLayer())
            self.lstm.append(LstmLayer(depth=1, channel_in=self.channels[i] * 2 + self.channels[i+1],
                                       channel_hidden=self.channels[i],
                                       kernel_size=self.kernel,
                                       dropout=self.dropout,
                                       rec_dropout=self.dropout,
                                       peephole=self.peephole,
                                       gpu=self.gpu))


    def _init(self, batch, height, width):
        """
        Initialize all lists for the first iteration.
        
        batch := batch size of input
        height := height of input
        width := width of input
        """
        A = [[] for i in range(self.layer)]
        Ah = [[] for i in range(self.layer)]
        E = [[] for i in range(self.layer)]
        R = [[] for i in range(self.layer)]
        
        for i in range(self.layer):
            if self.gpu:
                E[i].append(torch.zeros(batch, self.channels[i] * 2,
                                        height // 2**i,
                                        width // 2**i).cuda())
                R[i].append(torch.zeros(batch, self.channels[i],
                                        height // 2**i,
                                        width // 2**i).cuda())
            else:
                E[i].append(torch.zeros(batch, self.channels[i] * 2,
                                        height // 2**i,
                                        width // 2**i))
                R[i].append(torch.zeros(batch, self.channels[i],
                                        height // 2**i,
                                        width // 2**i))
        
        return A, Ah, E, R

        
    def forward(self, x, mode='prediction'):
        """
        x should look like (T x B x C x H x W)
        
        todo: Adding multi-frame prediction.
        """
        # initialize all variables
        A, Ah, E, R = self._init(x.size(1), x.size(3), x.size(4))
        A[0] = x

        # loop through the time series
        for t in range(1, len(x)):
            # loop through the layer
            # top-down pass
            for l in range(self.layer - 1, -1, -1):
                # compute recurrences
                if l == self.layer - 1:
                    R[l].append(self.lstm[l](E[l][t-1], R[l][t-1])[0][-1])
                else:
                    R[l].append(self.lstm[l](torch.cat((E[l][t-1],
                                f.interpolate(R[l+1][t], scale_factor=2,
                                              mode='nearest')), dim=1),
                                R[l][t-1])[0][-1])
            # forward pass
            for l in range(self.layer):
                # compute predictions
                if l == 0:
                    # SatLU (saturation linear unit)
                    Ah[l].append(f.hardtanh(self.prediction[l](R[l][t]), 0,
                                            self.pixel_max))
                else:
                    Ah[l].append(self.prediction[l](R[l][t]))
                # compute errors
                E[l].append(self.error[l](A[l][t-1], Ah[l][t-1]))

                if l < self.layer - 1:
                    A[l+1].append(self.input[l](E[l][t]))

        if mode == 'prediction':
            return Ah[0]
        elif mode == 'error':
            return E[0]
        else:
            raise IOError('No valid option given, please use: <prediction|error>')
