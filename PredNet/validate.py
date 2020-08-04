#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 30.05.2020
@author: Sören S. Dittrich
@version: 0.0.2
@description: Validation file
""" 

import torch

from helper.transformation import normalize, binarize
from helper.loss import loss as Loss


def validation(name, model, lossp, dataloader, logger, device, norm, binar):
    """
    Validate the model
    
    name := name of the model
    model := initialized network model
    lossp := name of loss to use <mae|mse|bce|bcel>
    dataloader := initialized dataloader
    logger := initialized tensorboard logger
    device := GPU or CPU
    norm := True if normalization is required, False otherwise
    binar := True if binarization is required, False otherwise
    """
    # set model in evaluation mode
    model.eval()
    logger.set_mode('validate')
    bloss = 0.0

    it = 0

    # run through all batches in the dataloader
    for batch_id, data in enumerate(dataloader):
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
            output = model(x[:-1])
            loss = Loss(output, x[-1], lossp)

        bloss += loss

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
            if lossp == 'bcel':
                logger.plot_image('predicted', torch.sigmoid(output[0]))
            else:
                logger.plot_image('predicted', output[0])

    return bloss / it