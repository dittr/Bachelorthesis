#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 13.06.2020
@author: Soeren Dittrich
@version: 0.0.1
@description: PredRNN module (ST-LSTM and PredRNN structure)
"""

import torch
import torch.nn as nn
import numpy as np


def pad_size(kernel):
    """
    """
    return int(np.floor(kernel / 2))


class ST_LSTM_CELL(nn.Module):
    """
    PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs,
    by Wang et. al.
    """
    def __init__(self, channel_in, channel_hidden, kernel, padding,
                 dropout=None, rec_dropout=None):
        """
        todo: add dropout layer
        """
        super(ST_LSTM_CELL, self).__init__()

        # input gate
        self.w_xi = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False)
        self.w_hi = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)
        self.w_xid = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_hidden,
                               kernel_size=kernel,
                               padding=padding,
                               bias=False)
        self.w_mi = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)

        # forget gate
        self.w_xf = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False)
        self.w_hf = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)
        self.w_xfd = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_hidden,
                               kernel_size=kernel,
                               padding=padding,
                               bias=False)
        self.w_mf = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)

        # output gate
        self.w_xo = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False)
        self.w_ho = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)
        self.w_co = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False)
        self.w_mo = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False)

        # cell state
        self.w_xg = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False)
        self.w_hg = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)
        self.w_xgd = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_hidden,
                               kernel_size=kernel,
                               padding=padding,
                               bias=False)
        self.w_mg = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel,
                              padding=padding,
                              bias=True)

        # 1x1 layer
        self.w_xx = nn.Conv2d(in_channels=channel_hidden * 2,
                              out_channels=channel_hidden,
                              kernel_size=1,
                              padding=0,
                              bias=False)


    def forward(self, x, h_old, c_old, m_old):
        """
        * is the hadamard product
        """
        # standard temporal memory
        g = torch.tanh(self.w_xg(x) + self.w_hg(h_old))
        i = torch.sigmoid(self.w_xi(x) + self.w_hi(h_old))
        f = torch.sigmoid(self.w_xf(x) + self.w_hf(h_old))
        c = f * c_old + i * g

        # spatio-temporal memory
        g_ = torch.tanh(self.w_xgd(x) + self.w_mg(m_old))
        i_ = torch.sigmoid(self.w_xid(x) + self.w_mi(m_old))
        f_ = torch.sigmoid(self.w_xfd(x) + self.w_mf(m_old))
        m = f_ * m_old + i_ * g_

        o = torch.sigmoid(self.w_xo(x) + self.w_ho(h_old) +
                          self.w_co(c) + self.w_mo(m))
        h = o * torch.tanh(self.w_xx(torch.cat((c, m), dim=1)))

        return h, c, m


class PredRNN(nn.Module):
    """
    PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs,
    by Wang et. al.
    """
    def __init__(self, depth, channel_in, channel_hidden, kernel_size,
                 dropout=None, rec_dropout=None, gpu=False):
        """
        """
        super(PredRNN, self).__init__()

        self.lstm = nn.ModuleList()
        self.dim = list()
        padding = list()
        self.gpu = gpu

        # used for convenience
        if type(channel_hidden) == list:
            channel_in = [channel_in] + channel_hidden
        else:
            channel_in = [channel_in] + [channel_hidden]
        if type(kernel_size) == int:
            kernel_size = [kernel_size for i in range(depth)]

        for i in range(len(kernel_size)):
            padding.append(pad_size(kernel=kernel_size[i]))

        for i in range(depth):
            if depth == 1:
                self.dim.append(channel_hidden)
            else:
                self.dim.append(channel_hidden[i])
            
            self.lstm.append(ST_LSTM_CELL(channel_in=channel_in[i],
                                             channel_hidden=channel_in[i+1],
                                             kernel=kernel_size[i],
                                             padding=padding[i],
                                             dropout=dropout,
                                             rec_dropout=rec_dropout))


    def _init_hidden(self, batch_size, channel, height, width):
        """
        """
        if self.gpu:
            h = torch.zeros(batch_size, channel, height, width).cuda()
            c = torch.zeros(batch_size, channel, height, width).cuda()
            m = torch.zeros(batch_size, channel, height, width).cuda()
        else:
            h = torch.zeros(batch_size, channel, height, width)
            c = torch.zeros(batch_size, channel, height, width)
            m = torch.zeros(batch_size, channel, height, width)

        return h, c, m


    def forward(self, x, h=None, c=None, m=None):
        """
        """
        if len(x.size()) != 5:
            x = x[None,:,:,:,:]

        time = len(x)
        depth = len(self.lstm)
        
        # 1. Loop through the stages / depth
        for i in range(depth):
            h_out, c_out, m_out = list(), list(), list()
            # 2. Initialize cell-state
            _, bsize, _, height, width = x.size()
            if type(h) == type(None) and type(c) == type(None) \
            and type(m) == type(None):
                h, c, m = self._init_hidden(bsize, self.dim[i],
                                            height, width)
            elif type(h) == type(None) and type(c) == type(None):
                h, c, _ = self._init_hidden(bsize, self.dim[i],
                                            height, width)
            elif type(c) == type(None):
                _, c, m = self._init_hidden(bsize, self.dim[i],
                                            height, width)
            elif type(m) == type(None):
                _, _, m = self._init_hidden(bsize, self.dim[i],
                                            height, width)
            # 3. Loop through the sequence
            for j in range(time):
                h, c, m = self.lstm[i].forward(x[j], h, c, m)          
                h_out.append(h)
                c_out.append(c)
                m_out.append(m)
                
            x = torch.cat(h_out)[:,None,:,:,:].reshape(time,
                         bsize, -1, height, width)
            h = None
            c = None
        
        # return prediction and cell-state
        return h_out[-1], (h_out[-1], c_out[-1], m_out[-1])
