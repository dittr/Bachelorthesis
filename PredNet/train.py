#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 29.05.2020
@author: Sören S. Dittrich
@version: 0.0.3
@description: Train file
"""

from validate import validation

from helper.transformation import normalize, binarize
from helper.loss import loss as Loss

import matplotlib.pyplot as plt


def train(model, optim, schedule, lossp, dataloader, device, logger,
          epoch, save, validate, depoch, diteration, norm, binar, debug=False):
    if debug:
        print('[DEBUG] Start training.')

    it = 0

    # run through the epochs, omit some if pre-trained
    for i in range(depoch, epoch):
        # set model in training mode
        model.train()
        logger.set_mode('training')
        bloss = 0.0
        
        # run through all batches in the dataloader
        for batch_id, (data, target) in enumerate(dataloader[0]):
            # get sequence and target (This is only for gray-scaled images.)
            x = data.float().to(device)[:,:,None,:,:].permute(1,0,2,3,4)
            y = target.float().to(device)
            if norm or binar:
                x = normalize(x)
                y = normalize(y)
            if binar:
                x = binarize(x)
                y = binarize(y)

            # clear optimizer
            optim.zero_grad()

            # forward pass
            output = model(x)

            # compute loss
            loss = Loss(output[-1], y, lossp)
            
            bloss += loss.item()

            # plot output and y
            plt.imsave('out.png', output[-1][0][0].cpu().detach())
            plt.imsave('true.png', y[0][0].cpu())

            # backpropagation
            loss.backward()

            # update weights
            optim.step()

            it += 1

            # log scalar values
            logger.plot_loss(lossp, loss.item(), it)

        # validation every n epochs
        if (i+1) % validate == 0:
            if len(dataloader) > 1:
                validation(model, lossp, dataloader[1], logger, device,
                           normalize, binarize)
            else:
                print('Test Epoch: {} Mean loss: {:.6f}'.format(i+1, bloss / batch_id))

        # perform scheduler update
        schedule.step()
