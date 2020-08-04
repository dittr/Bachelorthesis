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
    val_file = 'moving_mnist_val.hkl'
    
    def __init__(self, root, seq_len, testing=False, val=False):
        """
        """
        self.root = root
        self.seq_len = seq_len
        self.testing = False
        self.val = val
        
        if self.val:
            self.val_data = torch.from_numpy(hkl.load(os.path.join(self.root,
                                                                   self.val_file)))
        else:
            if not self.testing:  
                self.train_data = torch.from_numpy(hkl.load(os.path.join(self.root,
                                                                         self.training_file)))
            else:
                self.test_data = torch.from_numpy(hkl.load(os.path.join(self.root,
                                                                        self.test_file)))
        
        
    def __getitem__(self, index):
        """
        """
        if self.val:
            return self.val_data[self.seq_len*index:self.seq_len*(index+1)]
        
        if not self.testing:
            return self.train_data[self.seq_len*index:self.seq_len*(index+1)]
        else:
            return self.test_data[self.seq_len*index:self.seq_len*(index+1)]


    def __len__(self):
        """
        """
        if self.val:
            return len(self.val_data) // self.seq_len
        
        if not self.testing:
            return len(self.train_data) // self.seq_len
        else:
            return len(self.test_data) // self.seq_len
