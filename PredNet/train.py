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
from helper.loss import error_loss as Loss


def train(model, optim, schedule, lossp, dataloader, device, logger,
          epoch, iteration, save, validate, depoch, diteration, norm, binar,
          time_weight, layer_weight, debug=False):
    """
    Training the model
    
    model := initialized network model
    optim := initialized optimizer
    schedule := initialized lr scheduler
    lossp := name of loss to use <mae|mse|bce|bcel>
    dataloader := initialized dataloader
    device := GPU or CPU
    logger := initialized tensorboard logger
    epoch := epochs to run
    iteration := iterations to run per epoch
    save := True if model should be saved, False otherwise
    validate := validate performance after n epochs
    depoch := already performed amount of epochs (Done epoch)
    diteration := already performed amount of iterations (Done iteration)
    norm := True if normalization is required, False otherwise
    binar := True if binarization is required, False otherwise
    time_weight := initialized list of time weights
    layer_weight := initialized list of layer weights
    debug := debug
    """
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
        for batch_id, data in enumerate(dataloader[0]):
            # get sequence and target
            x = data.to(device).float().permute(1,0,2,3,4)
            if norm or binar:
                x = normalize(x)
            if binar:
                x = binarize(x)

            # clear optimizer
            optim.zero_grad()

            # forward pass
            output = model(x)

            # compute loss
            loss = Loss(time_weight, layer_weight, len(time_weight), output)
            
            bloss += loss

            # backpropagation
            loss.backward()

            # update weights
            optim.step()

            it += 1

            # log scalar values
            logger.plot_loss('error', loss, it)

            if batch_id >= iteration and iteration > 0:
                break

        # validation every n epochs
        if (i+1) % validate == 0:
            if len(dataloader) > 1:
                validation(model, lossp, dataloader[1], logger, device,
                           normalize, binarize)
            else:
                print('Epoch: {} Mean loss: {:.6f}'.format(i+1, bloss / (batch_id + 1)))

        if i > 0:
            # perform scheduler update
            schedule.step()
