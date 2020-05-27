#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 27.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Save and load network
""" 

from dataclasses import dataclass
from collections import OrderedDict
import torch

@dataclass
class ModelParameter:
    epoch: int
    iteration: int
    mdl_state: OrderedDict
    optim_state: OrderedDict
    loss: float


class ModelSaver():
    """
    Class to save a model to a certain path
    """
    def __init__(self, dataset, path='mdl/', mdl='prednet', debug=False):
        """
        """
        self.dataset = dataset
        self.path = path 
        self.mdl = mdl
        self.debug

    def save(self, param):
        """
        """
        torch.save({
                'epoch': param.epoch,
                'iteration': param.iteration,
                'model_state': param.mdl_state,
                'optimizer_state': param.optim_state,
                'loss': param.loss,
                }, self.path + self.mdl + '_' + self.dataset + '.pth')
        
        if self.debug:
            print('[ModelSaver] Model saved.')


class ModelLoader():
    """
    Class to load a model from a certain path
    """
    def __init__(self, dataset, path='mdl/', mdl='prednet', debug=False):
        """
        """
        self.dataset = dataset
        self.path = path
        self.mdl = mdl
        self.debug = debug

    def load(self):
        """
        """
        param = ModelParameter
        checkp = torch.load(self.path + self.mdl + '_' +
                            self.dataset + '.pth')
        
        param.epoch = checkp['epoch']
        param.iteration = checkp['iteration']
        param.mdl_state = checkp['model_state']
        param.optim_state = checkp['optimizer_state']
        param.loss = checkp['loss']
        
        return param
        