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
    def __init__(self, dataset, path='mdl/', mdl='prednet',
                 debug=False, logger=None):
        """
        Initialization
        
        dataset := used dataset
        path := path to save the model
        mdl := name of the model
        debug := print to stdout and to logger if True
        logger := logger object
        """
        self.dataset = dataset
        self.path = path 
        self.mdl = mdl
        self.debug = debug
        self.logger = logger

    def save(self, param):
        """
        Save model to file
        
        param := ModelParameter class with values from trained model
        """
        file = self.path + self.mdl + '_' + self.dataset + '.pth'
        torch.save({
                'epoch': param.epoch,
                'iteration': param.iteration,
                'model_state': param.mdl_state,
                'optimizer_state': param.optim_state,
                'loss': param.loss,
                }, file)

        if self.debug:
            deb = '[ModelSaver] Epoch: {}, Iteration: {}, State: {}, Optim: {},\
                  Loss: {}, File: {}'.format(param.epoch, param.iteration,
                                             param.mdl_state, param.optim_state,
                                             param.loss, file)
            print(deb)
            if type(self.logger) != None:
                self.logger.write(deb)


class ModelLoader():
    """
    Class to load a model from a certain path
    """
    def __init__(self, dataset, device, path='mdl/', mdl='prednet',
                 debug=False, logger=None):
        """
        Initialization
        
        dataset := the dataset to use
        path := path to the saved model
        mdl := name of the model
        debug := print to stdout and to logger if True
        logger := logger object
        """
        self.dataset = dataset
        self.device = device
        self.path = path
        self.mdl = mdl
        self.debug = debug
        self.logger = logger

    def load(self):
        """
        Load model from file
        """
        param = ModelParameter
        file = self.path + self.mdl + '_' + self.dataset + '.pth'
        checkp = torch.load(file, map_location=self.device)

        param.epoch = checkp['epoch']
        param.iteration = checkp['iteration']
        param.mdl_state = checkp['model_state']
        param.optim_state = checkp['optimizer_state']
        param.loss = checkp['loss']
        
        if self.debug:
            deb = '[ModelSaver] Epoch: {}, Iteration: {}, State: {}, Optim: {},\
                  Loss: {}, File: {}'.format(param.epoch, param.iteration,
                                             param.mdl_state, param.optim_state,
                                             param.loss, file)
            print(deb)
            if type(self.logger) != None:
                self.logger.write(deb)

        return param
