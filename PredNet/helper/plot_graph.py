#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 04.06.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Plotting backpropagation graph module
"""

import torch
import os

from torchviz import make_dot
from graphviz import Source

def plot_graph(model, size, path, gpu):
    """
    Plotting the backprop graph

    model := initialized model
    size := size of input image
    path := Where to plot the graph
    gpu := GPU or CPU
    """
    if gpu:
        x = torch.randn(1,1,1,size[0],size[1]).cuda()
    else:
        x = torch.randn(1,1,1,size[0],size[1])

    dot_graph = make_dot(model(x), params=dict(model.named_parameters()))
    Source(dot_graph).render(os.path.join(path, 'prednet.jpg'))
