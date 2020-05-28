#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Validation file
""" 

import torch

from helper.loss import loss as Loss

def validation(model, lossp, dataloader, logger, device):
    # set model in evaluation mode
    model.eval()
    logger.set_mode('validate')

    it = 0

    # run through all batches in the dataloader
    for batch_id, (data, target) in enumerate(dataloader):
        # get sequence and target (This is only for gray-scaled images.)
        x = data.float().to(device)[:,:,None,:,:].permute(1,0,2,3,4)
        y = target.float().to(device)[:,:,None,:,:].permute(1,0,2,3,4)

        # forward pass
        output = model(x)

        # compute loss
        loss = Loss(output[-1], y, lossp)

        it += 1

        # log scalar values
        logger.plot_loss(loss.item(), it)

        # log images
        x = torch.cat((x[1:], y), dim=0).permute(1,0,2,3,4)[0]
        output = output.permute(1,0,2,3,4)[0]
        
        logger.plot_images('ground_truth', x)
        logger.plot_images('predicted', output)
