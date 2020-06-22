#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 04.06.2020
@author: Sören S. Dittrich
@version: 0.0.4
@description: PredNet module
"""

#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as f

from model.modules.input import InputLayer
from model.modules.prediction import PredictionLayer
from model.modules.error import ErrorLayer
from model.modules.conv_lstm import CONV_LSTM
from model.modules.predrnn import PredRNN

class PredNet(nn.Module):
    """
    """
    def __init__(self, channels, kernel, padding, stride,
                 dropout, peephole, pixel_max, mode,
                 predrnn=True, extrapolate=0, gpu=False):
        """
        Initialize PredNet module
        
        channels := list of channels
        kernel := kernel size for all conv layers
        padding := padding to use for all conv layers
        stride := stride for max-pooling
        dropout := value for dropout [0,1]
        peephole := use ConvLSTM with or without peephole
        pixel_max := max value of input image
        mode := Mode for PredNet <prediction|error>
        extrapolate := Multi-frame prediction t+n into future
        gpu := GPU or CPU
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
        self.extrapolate = extrapolate
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
            if not predrnn:
                self.lstm.append(CONV_LSTM(depth=1, channel_in=self.channels[i] \
                                           * 2 + self.channels[i+1],
                                           channel_hidden=self.channels[i],
                                           kernel_size=self.kernel,
                                           dropout=self.dropout,
                                           rec_dropout=self.dropout,
                                           peephole=self.peephole,
                                           gpu=self.gpu))
            else:
                self.lstm.append(PredRNN(depth=1, channel_in=self.channels[i] \
                                         * 2 + self.channels[i+1],
                                         channel_hidden=self.channels[i],
                                         kernel_size=self.kernel,
                                         dropout=self.dropout,
                                         rec_dropout=self.dropout,
                                         gpu=self.gpu))
        self.upsample = nn.Upsample(scale_factor=2)


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
        Forward method
        
        x := input timeseries
        """
        # initialize all variables
        A, Ah, E, R, H = self._init(x.size(1), x.size(3), x.size(4))
        error = list()
        A[0] = x
        output = list()
        t, i = 0, 0

        # loop through the time series
        while t < len(x):
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
                                self.upsample(R[l+1][t+1])), dim=1),
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
#                print('Layer: ' + str(l+1) + ' Values: ' + str(Ah[l][-1].max()))
                # compute errors
                E[l].append(self.error[l](Ah[l][t], A[l][t]))
#                print('Layer: ' + str(l+1) + ' Error: ' + str(E[l][-1]))
#                plt.imshow(Ah[0][t][0][0].detach().cpu())
#                plt.show()
                # using MovingMNIST needs check when starting with normalized images
#                if Ah[0][t][0][0].max() == 0:
#                    raise Exception('Retry, because network will not learn!')
                
                if l < self.layer - 1:
                    A[l+1].append(self.input[l](E[l][t+1]))

            if self.mode == 'error':              
                mean_error = torch.cat([torch.mean(e[-1].view(e[-1].size(0), -1), 1,
                                                   keepdim=True) for e in E], 1)
                error.append(mean_error)

            if self.extrapolate > i and t == (len(x)-1):
                if i == 0:
                    output.append(Ah[0])
                else:
                    output.append(Ah[0][-1])
                A[0] = torch.cat((A[0][1:], Ah[0][-1][None,:,:,:,:]))
                i += 1
                t = -1
                Ah[0] = list()

            t += 1

        if self.mode == 'prediction':
            if len(output) > 0:
                return output
            else:
                return Ah[0]
        elif self.mode == 'error':
            error = torch.stack(error, 2)
            return error
        else:
            raise IOError('No valid option given, please use: <prediction|error>')
