 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 26.05.2020
@author: Soeren Dittrich
@version: 0.0.1
@description: Convolutional LSTM module
"""

import torch
import torch.nn as nn
from model.modules.utils import pad_size


class CONV_LSTM_CELL(nn.Module):
    """
    Convolutional LSTM cell without peephole connections
    """
    def __init__(self, channel_in, channel_hidden, kernel, padding,
                 dropout=None, rec_dropout=None):
        """
        """
        super(CONV_LSTM_CELL, self).__init__()
        
        # input gate
        self.w_xi = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_hi = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        
        # forget gate
        self.w_xf = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_hf = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        
        # cell-state
        self.w_xc = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_hc = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        
        # output gate
        self.w_xo = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_ho = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        
        # Boolean if hidden alredy set for the cell
        self.init = False
        
        if dropout != None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = nn.Dropout2d(0)
            
        if rec_dropout != None:
            self.rec_dropout = nn.Dropout2d(rec_dropout)
        else:
            self.rec_dropout = nn.Dropout2d(0)


    def forward(self, x, h_old, c_old):
        """
        """
        # input gate            
        i = torch.sigmoid(self.w_xi(self.dropout(x)) +
                          self.w_hi(self.rec_dropout(h_old)))
        
        # forget gate
        f = torch.sigmoid(self.w_xf(self.dropout(x)) +
                          self.w_hf(self.rec_dropout(h_old)))
        
        # cell-state 
        c_t = torch.tanh(self.w_xc(self.dropout(x)) +
                         self.w_hc(self.rec_dropout(h_old)))
        c = c_t * i + c_old * f
        
        # output
        o = torch.sigmoid(self.w_xo(self.dropout(x)) +
                          self.w_ho(self.rec_dropout(h_old)))
        h = o * torch.tanh(c)
        
        # Return output and cell-state
        return h, c


class CONV_LSTM_PEEPHOLE_CELL(nn.Module):
    """
    Convolutional LSTM cell with peephole connection
    """
    def __init__(self, channel_in, channel_hidden, kernel, padding,
                 dropout=None, rec_dropout=None):
        """
        """
        super(CONV_LSTM_PEEPHOLE_CELL, self).__init__()
        
        # input gate
        self.w_xi = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_hi = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        self.w_ci = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        
        # forget gate
        self.w_xf = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_hf = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        self.w_cf = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        
        # cell-state
        self.w_xc = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_hc = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        
        # output gate
        self.w_xo = nn.Conv2d(in_channels=channel_in,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        self.w_ho = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=True)
        self.w_co = nn.Conv2d(in_channels=channel_hidden,
                              out_channels=channel_hidden,
                              kernel_size=kernel, padding=padding,
                              bias=False)
        
        # Boolean if hidden alredy set for the cell
        self.init = False

        if dropout != None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = nn.Dropout2d(0)
            
        if rec_dropout != None:
            self.rec_dropout = nn.Dropout2d(rec_dropout)
        else:
            self.rec_dropout = nn.Dropout2d(0)


    """
    """
    def forward(self, x, h_old, c_old):
        # input gate            
        i = torch.sigmoid(self.w_xi(self.dropout(x)) +
                          self.w_hi(self.rec_dropout(h_old)) +
                          self.w_ci(c_old))
        
        # forget gate
        f = torch.sigmoid(self.w_xf(self.dropout(x)) +
                          self.w_hf(self.rec_dropout(h_old)) +
                          self.w_cf(c_old))
        
        # cell-state 
        c_t = torch.tanh(self.w_xc(self.dropout(x)) +
                         self.w_hc(self.rec_dropout(h_old)))
        c = c_t * i + c_old * f
        
        # output
        o = torch.sigmoid(self.w_xo(self.dropout(x)) +
                          self.w_ho(self.rec_dropout(h_old)) +
                          self.w_hc(c))
        h = o * torch.tanh(c)
        
        # Return output and cell-state
        return h, c


class CONV_LSTM(nn.Module):
    """
    Convolutional LSTM module
    """
    def __init__(self, depth, channel_in,
                 channel_hidden, kernel_size,
                 dropout=None, rec_dropout=None,
                 peephole=True, gpu=False):
        """
        """
        super(CONV_LSTM, self).__init__()
        
        self.lstm = nn.ModuleList()
        self.dim = list()
        padding = list()
        self.gpu = gpu
        self.dim = channel_hidden
        
        # used for convenience
        channel_in = [channel_in] + channel_hidden
        if type(kernel_size) == type(int):
            kernel_size = [kernel_size for i in range(depth)]
        
        for i in range(len(kernel_size)):
            padding.append(pad_size(kernel=kernel_size[i]))
        
        # Create deep ConvLSTM module
        for i in range(depth):
            self.dim.append(channel_hidden[i])
            
            if not peephole:
                self.lstm.append(CONV_LSTM_CELL(channel_in=channel_in[i],
                                                channel_hidden=channel_in[i+1],
                                                kernel=kernel_size[i],
                                                padding=padding[i],
                                                dropout=dropout,
                                                rec_dropout=rec_dropout))
            else:
                self.lstm.append(CONV_LSTM_PEEPHOLE_CELL(channel_in=channel_in[i],
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
        else:
            h = torch.zeros(batch_size, channel, height, width)
            c = torch.zeros(batch_size, channel, height, width)
            
        return h, c
    

    def forward(self, x, h=None, c=None):
        """
        """
        time = len(x)
        depth = len(self.lstm)
        
        # 1. Loop through the stages / depth
        for i in range(depth):
            h_out, c_out = list(), list()
            # 2. Initialize cell-state
            _, bsize, _, height, width = x.size()
            if type(h) == type(None) and type(c) == type(None):
                h, c = self._init_hidden(bsize, self.dim[i],
                                         height, width)
            elif type(c) == type(None):
                _, c = self._init_hidden(bsize, self.dim[i],
                                         height, width)
            # 3. Loop through the sequence
            for j in range(time):
                h, c = self.lstm[i].forward(x[j], h, c)
                h_out.append(h)
                c_out.append(c)
                
            x = torch.cat(h_out)[:,None,:,:,:].reshape(time,
                         bsize, -1, height, width)
            h = None
            c = None
        
        # return prediction and cell-states
        return h_out, c_out
