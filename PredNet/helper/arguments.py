#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 27.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Console arguments
"""

import argparse


class ConsoleArguments():
    """
    """
    def __init__(self):
        """
        """
        parser = argparse.ArgumentParser(description='Re-implementation of \
                                         PredNet using PyTorch.')

        parser.add_argument('--parameter', '-p', type=str,
                            help='Path to yml parameter file')
        parser.add_argument('--dataset', '-d', type=str,
                            help='mnist, face, kitti')
        parser.add_argument('--epoch', '-e', type=int,
                            help='Number of epochs')
        parser.add_argument('--batch', '-b', type=int,
                            help='Batchsize')
        parser.add_argument('--sequence', '-s', type=int,
                            help='Length of input sequence')
        parser.add_argument('--testing', '-t', action='store_true',
                            help='Set mode to testing (Default: False)')
        parser.add_argument('--optimizer', '-o', type=str,
                            help='Name of optimzer to use <adam|rmsprop>')
        
        
        self.args = parser.parse_args()

    def get_parameter(self):
        """
        Return yml file path
        """
        return self.args.parameter

    def get_dataset(self):
        """
        Return name of dataset
        """
        return self.args.dataset

    def get_epoch(self):
        """
        Return number of epochs
        """
        return self.args.epoch

    def get_batch(self):
        """
        Return batch size
        """
        return self.args.batch

    def get_sequence(self):
        """
        Return sequence length
        """
        return self.args.sequence
    
    def get_testing(self):
        """
        Return if testing or training is required
        """
        return self.args.testing

    def get_optimizer(self):
        """
        Return the optimizer to use
        """
        return self.args.optimizer