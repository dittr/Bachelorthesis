#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 30.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: MovingMNIST dataset for PredNet
"""

import os
import torch
import torch.utils.data as data
import hickle as hkl


class MovingMNIST(data.Dataset):
    """
    """
    
    training_file = 'moving_mnist_train.hkl'
    test_file = 'moving_mnist_test.hkl'
    
    def __init__(self, root, seq_len, testing=False):
        """
        """
        self.root = root
        self.seq_len = seq_len
        self.testing = False        
        self.train_data = torch.from_numpy(hkl.load(os.path.join(self.root,
                                                                 self.training_file)))
        self.test_data = torch.from_numpy(hkl.load(os.path.join(self.root,
                                                                self.test_file)))
        
        
    def __getitem__(self, index):
        """
        """
        if not self.testing:
            return self.train_data[self.seq_len*index:self.seq_len*(index+1)]
        else:
            return self.test_data[self.seq_len*index:self.seq_len*(index+1)]


    def __len__(self):
        """
        """
        if not self.testing:
            return len(self.train_data) // self.seq_len
        else:
            return len(self.test_data) // self.seq_len
