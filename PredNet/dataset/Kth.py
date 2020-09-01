#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 02.06.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: KTH dataset for PredNet
"""

import os
import torch
import torch.utils.data as data
import hickle as hkl


class Kth(data.Dataset):
    """
    """

    test_file = 'kth_test.hkl'
    training_file = 'kth_train.hkl'
    validation_file = 'kth_dev.hkl'

    def __init__(self, root, seq_len, testing=False, validation=False):
        """
        """
        self.root = root
        self.seq_len = seq_len
        self.testing = testing
        self.validation = validation
        
        if self.validation:
            self.validation_data = hkl.load(os.path.join(self.root,
                                                         self.validation_file))
            self.validation_data = self._create_tensor(self.validation_data)
        else:
            if not self.testing:
                self.train_data = hkl.load(os.path.join(self.root,
                                                        self.training_file))
                self.train_data = self._create_tensor(self.train_data)
            else:
                self.test_data = hkl.load(os.path.join(self.root, self.test_file))
                self.test_data = self._create_tensor(self.test_data)


    def __getitem__(self, index):
        """
        """
        if self.validation:
            return self.validation_data[index][:,None,:,:]
        
        if not self.testing:
            return self.train_data[index][:,None,:,:]
        else:
            return self.test_data[index][:,None,:,:]


    def __len__(self):
        """
        """
        if self.validation:
            return self.validation_data.size(0)
        
        if not self.testing:
            return self.train_data.size(0)
        else:
            return self.test_data.size(0)


    def _create_tensor(self, dataset):
        X = list()
        
        for i in range(len(dataset)):
            amount = len(dataset[i]['frames']) // self.seq_len
            
            for j in range(amount):
                X.append(dataset[i]['frames'][self.seq_len*j:self.seq_len*(j+1)])
        
        return torch.FloatTensor(X)
