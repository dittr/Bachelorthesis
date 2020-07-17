 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 15.07.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: ConvLSTM/PredRNN Autoencoder
"""

import torch.nn as nn

from model.modules.conv_lstm import CONV_LSTM
from model.modules.predrnn import PredRNN


class AutoENC(nn.Module):
    """
    """
    def __init__(self, depth, channel, kernel,
                 padding, predrnn=False, gpu=False):
        """
        """
        super(AutoENC, self).__init__()
        self.name = 'convlstm'
        
        self.depth = depth
        self.predrnn = predrnn
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for i in range(self.depth):
            if self.predrnn:
                self.encoder.append(PredRNN(depth=1, channel_in=channel[i],
                                            channel_hidden=channel[i+1],
                                            kernel_size=kernel,
                                            gpu=gpu))
                self.decoder.append(PredRNN(depth=1, channel_in=channel[-1 - i],
                                            channel_hidden=channel[-1 - (i+1)],
                                            kernel_size=kernel,
                                            gpu=gpu))
            else:
                self.encoder.append(CONV_LSTM(depth=1, channel_in=channel[i],
                                              channel_hidden=channel[i+1],
                                              kernel_size=kernel,
                                              gpu=gpu))
                self.decoder.append(CONV_LSTM(depth=1, channel_in=channel[-1 - i],
                                              channel_hidden=channel[-1 - (i+1)],
                                              kernel_size=kernel,
                                              gpu=gpu))


    def forward(self, x):
        """
        """    
        if self.predrnn:
            # 1. Encoder
            for i in range(self.depth):
                x, (h, c, m) = self.encoder[i](x)

            # 2. Decoder
            for i in range(self.depth):
                x, (h, c, m) = self.decoder[i](x)
        
        else:
            # 1. Encoder
            for i in range(self.depth):
                x, (h, c) = self.encoder[i](x)

            # 2. Decoder
            for i in range(self.depth):
                x, (h, c) = self.decoder[i](x)

        return x