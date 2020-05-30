#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 29.05.2020
@author: Soeren Dittrich
@version: 0.0.1
@description: Normalize and binarize the image
""" 

def normalize(x):
    """
    """
    x = x / 255.0
    return x

def binarize(x, threshold=0.5):
    """
    """
    x = x > threshold
    return x.float()