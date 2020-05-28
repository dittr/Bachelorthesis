#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.2
@description: Train file
"""

from validate import validation

from helper.loss import loss as Loss

def train(model, optim, lossp, dataloader, device, logger, epoch, save, validate,
          depoch=0, diteration=0, debug=False):
    if debug:
        print('[DEBUG] Start training.')

    it = 0

    # run through the epochs, omit some if pre-trained
    for i in range(depoch, epoch):
        # set model in training mode
        model.train()
        logger.set_mode('training')
        
        # run through all batches in the dataloader
        for batch_id, (data, target) in enumerate(dataloader[0]):
            # get sequence and target (This is only for gray-scaled images.)
            x = data.float().to(device)[:,:,None,:,:].permute(1,0,2,3,4)
            y = target.float().to(device)[:,:,None,:,:].permute(1,0,2,3,4)

            # clear optimizer
            optim.zero_grad()

            # forward pass
            output = model(x)
            
            # compute loss
            loss = Loss(output[-1], y, lossp)
            
            # backpropagation
            loss.backward()
            
            it += 1
            
            # log scalar values
            logger.plot_loss(loss.item(), it)
            
        # validation every n epochs
        if (i+1) % validate == 0:
            validation(model, lossp, dataloader[1], logger, device)
