#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 16.07.2020
@author: Soeren Dittrich
@version: 0.0.1
@description: Spatio-temporal Video Autoencoder implementation
"""

# necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# own imports
from model.modules.spatial_autoencoder import Encoder, Decoder
from model.modules.conv_lstm import CONV_LSTM as LSTM
from model.modules.predrnn import PredRNN
from model.modules.optical_flow import OpticalFlow
from model.modules.huber_penalty import HuberPenalty
from model.modules.grid_generator import GridGenerator

"""
"""
class AE_ConvLSTM_flow(nn.Module):
    """
    """
    def __init__(self, encoder, lstm, flow, huberp, grid, decoder,
                 forecast=1, gpu=False, trainable=False, training=False,
                 huber=False, reverse=False, predrnn=False):
        super(AE_ConvLSTM_flow, self).__init__()
        self.name = 'spatiotemp'
        self.forecast = forecast
        self.huber = huber
        self.reverse = reverse

        self.encoder = Encoder(depth_in=encoder['depth_in'],
                               depth_out=encoder['depth_out'],
                               kernel_size=encoder['kernel'])
        
        if not predrnn:
            self.lstm = LSTM(depth=1, channel_in=lstm['depth_in'],
                             channel_hidden=lstm['depth_out'],
                             kernel_size=lstm['kernel'],
                             gpu=gpu)
        else:
            self.lstm = PredRNN(channel_in=lstm['depth_in'],
                                channel_hidden=lstm['depth_out'],
                                kernel_size=lstm['kernel'],
                                gpu=gpu)
        
        self.flow = OpticalFlow(channel_in=flow['depth_in'],
                                channel_out=flow['depth_out'],
                                kernel_size=flow['kernel'],
                                clamp_min=-flow['clamp'],
                                clamp_max=flow['clamp'],
                                size=(flow['size']['y'],
                                      flow['size']['x']))
        
        self.huber = HuberPenalty(kernel_size=huberp['kernel'],
                                  mu=huberp['mu'],
                                  l1=huberp['l1'],
                                  gpu=gpu,
                                  training=training)
        
        self.grid = GridGenerator(size=(grid['size']['y'],
                                        grid['size']['x']),
                                  gpu=gpu)
        
        self.decoder = Decoder(depth_in=decoder['depth_in'],
                               depth_out=decoder['depth_out'],
                               kernel_size=decoder['kernel'],
                               scale=decoder['scale'])
        
        if not trainable:
            self._excl()
    
    """
    The weigths and biases of the huber loss are non-trainable
    """
    def _excl(self):
        self.huber.conv1.weight.requires_grad = False
        self.huber.conv2.weight.requires_grad = False
        self.huber.conv1.bias.requires_grad = False
        self.huber.conv2.bias.requires_grad = False

    """
    """
    def forward(self, x):
        enc = []
        output = []
        
        for i in range(self.forecast):
            # Encoder
            for i in range(x.size()[0]):
                enc.append(self.encoder(x[i]))
            
            # Save last image for image sampler
            x_l = enc[-1].clone().detach() # dont backprop over this
            
            # Sequence-to-Sequence like counter intuitive reversion of input
            if self.reverse:
                enc = enc[::-1]
        
            # ConvLSTM
            out, _ = self.lstm(torch.stack(enc))
        
            # Flow prediction
            out = self.flow(out[-1][None,:,:,:])
            flow = out.clone().detach()

            if self.huber:
                # Huber penalty
                out = self.huber(out)

            # Grid generator (normalization)
            out = self.grid(out).permute(0,2,3,1)
        
            # Image sampler
            out = F.grid_sample(x_l, out, mode='bilinear')
        
            # Decoder
            out = self.decoder(out)
            
            output.append(out)
            x = torch.cat([x[1:], out[None,:,:,:,:]], dim=0)
        
        return torch.stack(output)[0]
