#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 02.06.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Kitti dataset for PredNet
""" 

import os
import torch
import torch.utils.data as data
import hickle as hkl


class Kitti(data.Dataset):
    """
    """
    
    training_data = 'kitti_X_train.hkl'
    training_sources = 'kitti_sources_train.hkl'
    testing_data = 'kitti_X_test.hkl'
    testing_sources = 'kitti_sources_test.hkl'
    
    def __init__(self, root, seq_len, testing=False):
        """
        """
        self.root = root
        self.seq_len = seq_len
        self.testing = testing
        
        if not self.testing:
            self.train_data = hkl.load(os.path.join(self.root,
                                                    self.training_data))
            self.train_sources = hkl.load(os.path.join(self.root,
                                                       self.training_sources))
            self.train_data = self._create_tensor(self.train_data,
                                                  self.train_sources)
        else:
           self.test_data = hkl.load(os.path.join(self.root, self.testing_data))
           self.test_sources = hkl.load(os.path.join(self.root,
                                                     self.testing_sources))
           self.test_data = self._create_tensor(self.test_data,
                                                self.test_sources) 


    def __getitem__(self, index):
        """
        """
        if not self.testing:
            return self.train_data[index].permute(0,3,1,2)
        else:
            return self.test_data[index].permute(0,3,1,2)


    def __len__(self):
        """
        """
        if not self.testing:
            return self.train_data.size(0)
        else:
            return self.test_data.size(0)


    def _create_tensor(self, data, sources):
        """
        """
        X = list()

        for i in range(len(data) // self.seq_len):
            if len(set(sources[self.seq_len*i:self.seq_len*(i+1)])) == 1:
                X.append(data[self.seq_len*i:self.seq_len*(i+1)])

        return torch.FloatTensor(X)
        