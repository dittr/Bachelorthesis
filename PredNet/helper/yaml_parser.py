#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 26.05.2020
@author: Soeren Dittrich
@version: 0.0.1
@description: YAML file parser for network parameter
"""

import os
import yaml

def yml(path):
    if not os.path.exists(path):
        raise ValueError('File not available.')
        
    with open(path, 'r') as file:
        data = yaml.load(file, yaml.SafeLoader)
        
    return data 
