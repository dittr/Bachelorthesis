#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 30.05.2020
@author: Sören S. Dittrich
@version: 0.0.3
@description: Test file
""" 

import torch

from helper.transformation import normalize, binarize
from helper.loss import loss as Loss


def test(name, model, iteration, lossp, dataloader, logger, device, norm, binar):
    """
    Testing the model
    
    name := name of the model
    model := initialized network model
    iteration := iterations to run
    lossp := name of loss to use <mae|mse|bce|bcel>
    dataloader := initialized dataloader
    logger := initialized tensorboard logger
    device := GPU or CPU
    norm := True if normalization is required, False otherwise
    binar := True if binarization is required, False otherwise
    """
    # set model in evaluation mode
    model.eval()

    it = 0

    # run through all batches in the dataloader
    for batch_id, data in enumerate(dataloader):
        # get sequence and target (This is only for gray-scaled images.)
        x = data.to(device).float().permute(1,0,2,3,4)
        if norm or binar:
            x = normalize(x)
        if binar:
            x = binarize(x)

        # forward pass & compute loss
        if name == 'prednet':
            output = torch.stack(model(x))
            loss = Loss(output[1:len(x)], x[1:len(x)], lossp)
        else:
            output = model(x)
            loss = Loss(output, x[-1], lossp)

        it += 1

        # log scalar values
        logger.plot_loss(lossp, loss.item(), it)

        # log images
        x = x.permute(1,0,2,3,4)[0]
        logger.plot_images('ground_truth', x)
        
        if name == 'prednet':
            output = output.permute(1,0,2,3,4)[0]
            logger.plot_images('predicted', output)
        else:
            logger.plot_image('predicted', output[0])

        if batch_id >= iteration and iteration > 0:
            break
