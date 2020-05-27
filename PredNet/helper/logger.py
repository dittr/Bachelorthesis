#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 27.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Simple data logger
""" 

from dataclasses import dataclass
import os
import datetime


@dataclass
class Data:
    epoch: int
    iteration: int
    loss: float
    optimizer: str = ''
    loss_fct: str = ''


class DataLogger():
    """
    """
    def __init__(self, log_path='log/', mdl_name='prednet',
                 testing=False, verbose=False):
        """
        """
        self.testing = testing
        self.verbose = verbose

        # check if log_path already exists, create path if not
        self.path = os.path.join(log_path, mdl_name)

        if not os.path.exists(self.path):
            print('Creating path: ' + log_path + '/' + mdl_name)
            os.makedirs(self.path)

        date = str(datetime.date.today)
        time = str(datetime.datetime.now().hour) + \
               '-' + str(datetime.datetime.now().minute)

        try:
            self.file = open(self.path + '/[' + date + '-' + time + '].log', 'a')
        except IOError:
            print('[Log] Logger is not able to create log-file')
            self.logger = False


    def write(self, data):
        """
        """
        if not self.logger:
            return

        output = ''
        now = datetime.datetime.now()
        prefix = '[{}:{}:{}]'.format(now.hour, now.minute, now.second)

        if self.verbose:
            output = 'Epoch: {}, Iteration: {}, Loss: {}, Optimizer: {},\
                     Loss function: {}'.format(data.epoch, data.iteration,
                                               data.loss, data.optimizer,
                                               data.loss_fct)
        else:
            output = 'Epoch: {}, Iteration: {}, Loss: {}'.format(data.epoch,
                                                                 data.iteration,
                                                                 data.loss)

        try:
            self.file.write(prefix + ' ' + output)
        except IOError:
            print('[Log] File not existend or closed: logger stopped')
            self.logger = False
