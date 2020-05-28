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
    def __init__(self, channels, kernel, padding, dropout, peephole, gpu=False):
        """
        """
        super(PredNet, self).__init__()
        self.name = 'prednet'
        self.channels = channels
        self.kernel = kernel
        self.padding = padding
        self.dropout = dropout
        self.peephole = peephole
        self.gpu = gpu
        self.input = []
        self.prediction = []
        self.error = []
        self.lstm = []
        
        for i in range(self.layer - 1):
            self.input.append(InputLayer(channel_in=self.channels[i] * 2,
                                         channel_out=self.channels[i+1],
                                         kernel_size=self.kernel,
                                         padding=self.padding))
        
        for i in range(self.layer):
            self.prediction.append(PredictionLayer(channel_in=self.channels[i],
                                                   channel_out=self.channels[i],
                                                   kernel_size=self.kernel,
                                                   padding=self.padding))
            self.error.append(ErrorLayer())
            self.lstm.append(LstmLayer(depth=1, channel_in=self.channels[i] * 2,
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
                                        height, width).cuda())
                R[i].append(torch.zeros(batch, self.channels[i],
                                        height, width).cuda())
            else:
                E[i].append(torch.zeros(batch, self.channels[i] * 2,
                                        height, width))
                R[i].append(torch.zeros(batch, self.channels[i],
                                        height, width))
        
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
                if l == self.layer:
                    R[l].append(self.lstm[l](E[l][t-1], R[l][t-1]))
                else:
                    R[l].append(self.lstm[l](E[l][t-1], R[l][t-1],
                                f.interpolate(R[l+1][t],
                                              scale_factor=2,
                                              mode='nearest')))
            # forward pass
            for l in range(self.layer):
                # compute predictions
                if l == 0:
                    # SatLU (saturation linear unit)
                    Ah[l].append(torch.min(self.prediction[l](R[l][t]),
                                 self.pixel_max))
                else:
                    Ah[l].append(self.prediction[l](R[l][t]))
                # compute errors
                E[l][t].append(self.error(A[l][t], Ah[l][t]))

                if l < self.layer:
                    A[l+1][t] = self.input(E[l][t])

        if mode == 'prediction':
            return Ah[0]
        elif mode == 'error':
            return E[0]
        else:
            raise IOError('No valid option given, please use: <prediction|error>')
