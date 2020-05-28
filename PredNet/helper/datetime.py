#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Datetime module (Not mandatory)
""" 

import datetime

def get_today():
    """
    Simple wrapper to get todays value
    """
    return str(datetime.date.today())

def get_time():
    """
    Simple wrapper to get the time in format: hour_minute
    """
    return str(datetime.datetime.now().hour) + \
           '_' + str(datetime.datetime.now().minute)