#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 13.02.2020
@author: Soeren Dittrich
@version: 0.0.2
@description: Grid generator
"""

import numpy as np
import torch
import torch.nn as nn

"""
The flow prediction model already outputs a valid flow map, but it should be
normalized, to lay in the necessary realm of [-1, 1].
The top-left position is (-1, -1), bottom right is (1, 1).

--        ...       --
| [-1,-1] ... [-1,1] |
|                    |
| [1,-1   ... [1,1]  |
--        ...       --

"""
class GridGenerator(nn.Module):
    """
    """
    def __init__(self, size, gpu=False):
        super(GridGenerator, self).__init__()
        
        # Normalize the optical flow map, to lie in the pre defined range [-1,1]
        A1 = np.ones((1, size[0], size[1]))
        A2 = np.ones((1, size[1], size[0]))

        # x-dimension
        for i in range(size[0]):
            A1[0][i] = (-1 + (i)/(size[0]-1) * 2)
        # y-dimension
        for i in range(size[1]):
            A2[0][i] = (-1 + (i)/(size[1]-1) * 2)
        
        A1 = torch.from_numpy(A1)
        A2 = torch.from_numpy(A2.transpose(0,2,1))
        
        # create the normaliation map
        if gpu:
            self.A = torch.cat((A2, A1), 0).float().cuda()
        else:
            self.A = torch.cat((A2, A1), 0).float()

    """
    """
    def forward(self, x):
        return (x + self.A[None,:,:,:])
