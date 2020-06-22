#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.2
@description: Console arguments
"""

import argparse


class ConsoleArguments():
    """
    """
    def __init__(self):
        """
        Initialization of ArgumentParser
        """
        parser = argparse.ArgumentParser(description='Re-implementation of \
                                         PredNet using PyTorch.')

        parser.add_argument('--parameter', '-p', type=str,
                            help='Path to yml parameter file')
        parser.add_argument('--dataset', '-d', type=str,
                            help='Dataset to use <mnist|kth|kitti>')
        parser.add_argument('--epoch', '-e', type=int,
                            help='Number of epochs')
        parser.add_argument('--iteration', '-i', type=int,
                            help='Number of iterations per epoch')
        parser.add_argument('--batch', '-b', type=int,
                            help='Batchsize')
        parser.add_argument('--sequence', '-s', type=int,
                            help='Length of input sequence')
        parser.add_argument('--testing', '-t', action='store_true',
                            help='Set mode to testing (Default: False)')
        parser.add_argument('--optimizer', '-o', type=str,
                            help='Name of optimzer to use <adam|rmsprop>')
        parser.add_argument('--load', '-l', action='store_true',
                            help='Load model from mdl path (Default: False)')
        parser.add_argument('--save', '-S', action='store_true',
                            help='Save model to mdl path (Default: False)')
        parser.add_argument('--loss', '-L', type=str,
                            help='Loss function to use <mse|mae|bce|bcel|ssim>')
        parser.add_argument('--validate', '-v', type=int,
                            help='After how many epochs should start a validation.')
        parser.add_argument('--normalize', '-n', action='store_true',
                            help='Normalize the image data ({0,...,255} -> [0,1])')
        parser.add_argument('--binarize', '-B', action='store_true',
                            help='Binarize the image data ({0,1})')
        parser.add_argument('--learning_rate', '-r', type=float,
                            help='Learning rate for training.')
        parser.add_argument('--mode', '-m', type=str,
                            help='Mode for PredNet <prediction|error>')
        parser.add_argument('--plot', '-P', action='store_true',
                            help='Plot backpropagation graph (Default: False)')
        parser.add_argument('--extrapolate', '-E', type=int,
                            help='Extrapolate t+n images into future.')
        parser.add_argument('--predrnn', '-R', action='store_true',
                            help='False: ConvLSTM, True: ST_ConvLSTM (Default: False)')
        
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

    def get_iteration(self):
        """
        Return number of iterations per epoch
        """
        return self.args.iteration

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

    def get_load(self):
        """
        Return if use pre-trained model
        """
        return self.args.load

    def get_save(self):
        """
        Return if saving model
        """
        return self.args.save

    def get_loss(self):
        """
        Return the loss to use
        """
        return self.args.loss

    def get_validate(self):
        """
        Return the validation value
        """
        return self.args.validate

    def get_normalize(self):
        """
        Return the normalization value
        """
        return self.args.normalize

    def get_binarize(self):
        """
        Return the binarization value
        """
        return self.args.binarize

    def get_lr(self):
        """
        Return the learning rate value
        """
        return self.args.learning_rate

    def get_mode(self):
        """
        Return PredNet operating mode
        """
        return self.args.mode

    def get_plot(self):
        """
        Return plot value
        """
        return self.args.plot

    def get_extrapolate(self):
        """
        Return extrapolation value
        """
        return self.args.extrapolate

    def get_predrnn(self):
        """
        Return predrnn value
        """
        return self.args.predrnn
