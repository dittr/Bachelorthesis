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
                 dropout, peephole, pixel_max, mode, gpu=False):
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
        self.mode = mode
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
        H = [[] for i in range(self.layer)]
        
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
        
        return A, Ah, E, R, H

        
    def forward(self, x):
        """
        x should look like (T x B x C x H x W)
        
        todo: Adding multi-frame prediction.
        """
        # initialize all variables
        A, Ah, E, R, H = self._init(x.size(1), x.size(3), x.size(4))
        error = list()
        A[0] = x

        # loop through the time series
        for t in range(len(x)):
            # loop through the layer
            # top-down pass
            for l in range(self.layer - 1, -1, -1):
                if t == 0:
                    hx = (R[l][t], R[l][t])
                else:
                    hx = H[l]
                # compute recurrences
                if l == self.layer - 1:
                    Ra, hx = self.lstm[l](E[l][t], hx[0], hx[1])
                    R[l].append(Ra)
                else:
                    Ra, hx = self.lstm[l](torch.cat((E[l][t],
                                f.interpolate(R[l+1][t+1], scale_factor=2,
                                              mode='nearest')), dim=1),
                                hx[0], hx[1])
                    R[l].append(Ra)
                H[l] = hx

            # forward pass
            for l in range(self.layer):
                # compute predictions
                if l == 0:
                    # SatLU (saturation linear unit)
                    Ah[l].append(f.hardtanh(self.prediction[l](R[l][t+1]), 0,
                                            self.pixel_max))
                else:
                    Ah[l].append(self.prediction[l](R[l][t+1]))
                # compute errors
                E[l].append(self.error[l](A[l][t], Ah[l][t]))

                if l < self.layer - 1:
                    A[l+1].append(self.input[l](E[l][t+1]))
            
            if self.mode == 'error':
                mean_error = torch.cat([torch.mean(e[-1].view(e[-1].size(0), -1), 1,
                                                   keepdim=True) for e in E], 1)
                error.append(mean_error)

        if self.mode == 'prediction':
            return Ah[0]
        elif self.mode == 'error':
            error = torch.stack(error, 0).permute(1,2,0)
            return error
        else:
            raise IOError('No valid option given, please use: <prediction|error>')
