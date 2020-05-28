#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Tensorboard log module
"""

import os
from torch.utils.tensorboard import SummaryWriter

import helper.datetime as datetime

def TensorLog():
    """
    """
    def __init__(self, path, mdl_name, testing=False, debug=False):
        """
        """
        self.path = path
        self.mdl_name = mdl_name
        self.mode = 'training' if not testing else 'testing'
        self.debug = debug
        self.open = False
        self.date = datetime.get_today()
        self.time = datetime.get_time()

        if not os.path.exists(self.path):
            if self.debug:
                print('[DEBUG] Path not valid, create log path.')
            os.makedirs(self.path)

        self.path = os.path.join(self.path, self.mdl_name, self.date, self.time)
        if not os.path.exists(self.path):
            if self.debug:
                print('[DEBUG] Path not valid, create log path.')
            os.makedirs(self.path)
            
    def set_mode(self, value):
        """
        """
        self.mode = value

    def open_writer(self):
        self.writer = SummaryWriter(self.path)
        self.open = True
        
        if self.debug:
            print('[DEBUG] Opened SummaryWriter.')

    def is_open(self):
        """
        Get status of SummaryWriter
        """
        return self.open

    def write_text(self, text):
        """
        Write text into tensorboard log
        """
        # todo: implement

    def plot_loss(self, loss, it):
        """
        Plot loss values into tensorboard log
        
        loss := name of used loss
        it := current iteration
        """
        self.writer.add_scalar(self.mdl_name + '/' + self.mode + '/' + loss,
                               loss, it)

    def plot_image(self, name, image):
        """
        Plot one image into tensorboard log
        
        name := name where to save in log
        image := the image file
        """
        self.writer.add_image(self.mdl_name + '/' + name, image)

    def plot_images(self, name, images):
        """
        Plot n images in grid into tensorboard log
        
        name := name where to save in log
        image := the image file
        """
        self.writer.add_images(self.mdl_name + '/' + name, images)
    
    def close_writer(self):
        """
        Flush all values not written and close the writer
        """
        if self.open:
            self.writer.flush()
            self.writer.close()

        if self.debug:
            print('[DEBUG] Closed SummaryWriter.')