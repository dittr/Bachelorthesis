#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 04.06.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Caltech dataset for PredNet
"""

import os
import torch
import torch.utils.data as data
import hickle as hkl


class Caltech(data.Dataset):
    """
    """

    test_file = 'caltech_test.hkl'
    test_sources = 'caltech_test_sources.hkl'
    train_file = 'caltech_train.hkl'
    train_sources = 'caltech_train_sources.hkl'

    def __init__(self, root, seq_len, testing=False):
        """
        Initialize Caltech dataset
        
        root := where to find the hkl files
        seq_len := length of sequences
        testing := True if testing, False otherwise
        """
        self.root = root
        self.seq_len = seq_len
        self.testing = testing

        self.train_data = hkl.load(os.path.join(root, self.train_file))
        self.train_source_data = hkl.load(os.path.join(root, self.train_sources))
        self.test_data = hkl.load(os.path.join(root, self.test_file))
        self.test_source_data = hkl.load(os.path.join(root, self.test_sources))

        self.train_data = self._create_tensor(self.train_data,
                                              self.train_source_data)
        self.test_data = self._create_tensor(self.test_data,
                                             self.test_source_data)


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
