#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 26.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Convienience file to install all necessary libraries etc.
"""

from setuptools import setup

NAME = 'PredNet PyTorch'
DESCRIPTION = 'Re-implementation of PredNet using PyTorch'
AUTHOR = 'Sören S. Dittrich'
AUTHOR_EMAIL = 'dittr002@uni-hildesheim.de'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.0.1'

REQUIRED = [
    'numpy', 'matplotlib', 'torch', 'tensorboard', 'PyYAML'
]

with open('README') as f:
    readme = f.read()
    
setup(
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=readme,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      python_requires=REQUIRES_PYTHON,
      install_requires=REQUIRED
)
