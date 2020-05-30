#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.2
@description: Test file
""" 

import torch

from helper.transformation import normalize, binarize
from helper.loss import loss as Loss


def test(model, lossp, dataloader, logger, device, norm, binar):
    # set model in evaluation mode
    model.eval()

    it = 0

    # run through all batches in the dataloader
    for batch_id, data in enumerate(dataloader):
        # get sequence and target (This is only for gray-scaled images.)
        x = data.float().to(device)[:,:,None,:,:].permute(1,0,2,3,4)
        if norm or binar:
            x = normalize(x)
        if binar:
            x = binarize(x)

        # forward pass
        output = torch.stack(model(x))

        # compute loss
        loss = Loss(output, x, lossp)

        it += 1

        # log scalar values
        logger.plot_loss(lossp, loss.item(), it)

        # log images
        x = x.permute(1,0,2,3,4)[0]
        output = output.permute(1,0,2,3,4)[0]

        logger.plot_images('ground_truth', x)
        logger.plot_images('predicted', output)
