#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 28.05.2020
@author: Sören S. Dittrich
@version: 0.0.3
@description: Main-file
"""

# basic imports
import torch
import torch.optim as Optim

# net import
from model.prednet import PredNet

# datasets
from dataset.MovingMNIST import MovingMNIST
from dataset.Kitti import Kitti

# helper imports
from helper.arguments import ConsoleArguments
from helper.yaml_parser import yml
from helper.mdl import ModelLoader
from helper.mdl import ModelSaver
from helper.mdl import ModelParameter
from helper.tensorboard_log import TensorLog

# training and testing
from train import train
from test import test


def init_arguments():
    """
    Initialize the argparse module for console arguments.
    """
    return ConsoleArguments()

def init_yml(path):
    """
    Read yml file from console arguments
    
    path := path to yml file
    """
    return yml(path)

def init_device():
    """
    Initialize the device (GPU or CPU)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu = True
        
    return device, gpu
    
def init_model(channels, kernel, padding, stride, dropout,
               peephole, pixel_max, mode, gpu):
    """
    Initialize PredNet with arguments from yml file
    
    channels := list of channels
    kernel := kernel size
    padding := padding size
    stride := stride size (for max-pooling in input layer)
    dropout := percentage [0,1]
    peephole := LSTM using peephole
    pixel_max := maximum pixel value in input image
    mode := mode model uses <prediction|error>
    gpu := model uses gpu
    """   
    model = PredNet(channels, kernel, padding, stride, dropout,
                    peephole, pixel_max, mode, gpu)
    
    return model

def print_model(model):
    """
    Printing the models trainable parameter and the state_dict
    
    model := initialized network model
    """
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[DEBUG] ' + model.name + ' consists of: ' + str(trainable_param) +
      ' trainable parameters.')    
    print('[DEBUG] ' + model.state_dict())

def init_dataset(dataset, root, testing, download=True):
    """
    Initialize available datasets for training and validation or testing
    
    dataset := name of the dataset <mnist|kitti>
    testing := return testing set if True otherwise training set
    """
    data = []
    
    if dataset == 'mnist':
        data.append(MovingMNIST(root, testing, download=download))
    elif dataset == 'kitti':
        data.append(Kitti(root, testing))
    else:
        raise IOError('[ERROR] Choose a valid dataset <mnist|kitti>')
    
    return data

def get_dataloader(dataset, batch, shuffle, drop):
    """
    Return dataloader
    
    dataset := initialized dataset(s)
    batch := batch size
    shuffle := shuffle the dataset (True if training set, False otherwise)
    drop := necessary if size differs
    """
    dataloader = []
    
    for i in range(len(dataset)):
        if i == 0:
            dataloader.append(torch.utils.data.DataLoader(dataset=dataset[i],
                                                          batch_size=batch,
                                                          shuffle=shuffle,
                                                          drop_last=drop))
        else:
            dataloader.append(torch.utils.data.DataLoader(dataset=dataset[i],
                                                          batch_size=batch,
                                                          shuffle=not shuffle,
                                                          drop_last=drop))
    
    return dataloader

def init_optimizer(optim, model, lr):
    """
    Initialize the optimizer
    
    optim := name of optimizer <adam|rmsprop>
    model := initialized network model
    lr := learning rate
    """
    if optim == 'adam':
        optimizer = Optim.Adam(model.parameters(), lr)
    elif optim == 'rmsprop':
        optimizer = Optim.RMSProp(model.parameters(), lr)    
    else:
        raise IOError('[ERROR] Choose a valid optimizer <adam|rmsprop>')

    return optimizer

def init_scheduler(optimizer, step_size):
    """
    Initialize the learing rate scheduler
    
    optimizer := initialized optimizer
    step_size := reduce lr after step_size epochs
    """
    scheduler = Optim.lr_scheduler.StepLR(optimizer, step_size)
    
    return scheduler

def load_model(model, optimizer, device, dataset, path, debug=False):
    """
    Load the model if wanted
    
    model := initialized network model
    optimizer := initialized optimizer
    device := GPU or CPU
    dataset := name of used dataset <mnist|kitti>
    path := path from where to load the model
    debug := print debug output -> False
    """
    loader = ModelLoader(dataset, device, path, model.name, debug)
    params = loader.load()
    
    model.load_state_dict(params.mdl_state)
    optimizer.load_state_dict(params.optim_state)
    
    return model, params.epoch, params.iteration, optimizer, params.loss

def compute(testing, model, optimizer, scheduler, loss, dataloader,
            device, logger, epoch, depoch, diteration, save, validate,
            normalize, binarize, debug=False):
    """
    Compute test or training, given the flag testing
    
    testing := If true perform testing, otherwise training
    model := initialized network model
    optimizer := initialized optimizer
    scheduler := initialized scheduler
    loss := loss function to use
    dataloader := initialized dataloader
    device := GPU or CPU
    logger := tensorboard logger
    epoch := epochs to perform
    depoch := already performed epochs (Only pre-trained model)
    diteration := already performed iterations in peoch (Only pre-trained model)
    save := True if model should be saved, False otherwise
    validate := validate every n epochs
    normalize := normalize the image input
    binarize := binarize the image input
    debug := debug value
    """
    if not testing:
        train(model, optimizer, scheduler, loss, dataloader, device, logger,
              epoch, save, validate, depoch, diteration, normalize,
              binarize, debug)
    else:
        test(model, loss, dataloader[0], logger, device,
             normalize, binarize)

def save_model(model, optimizer, dataset, path, debug=False):
    """
    Save the model in the end
    
    model := initialized network model
    optimizer := initialized optimizer
    dataset := name of used dataset <mnist|kitti>
    path := path from where to load the model
    debug := print debug output -> False
    """
    saver = ModelSaver(dataset, path, model.name, debug)
    params = ModelParameter
    
    params.epoch = 0 # todo: change this!
    params.iteration = 0 # todo: change this!
    params.loss = 0 # todo: change this!
    params.mdl_state = model.state_dict()
    params.optim_state = optimizer.state_dict()
    
    saver.save(params)
    
def main():
    """
    Main file: Where the magic happens ;)
    """
    # 1. Initialize console arguments
    console = init_arguments()

    # 2. Initialize yml parameter
    args = init_yml(console.get_parameter())

    # 3. Check if yml parameter and console arguments are given (Console args are favored)
    # todo: implement

    # 4. Initialize device (GPU or CPU)
    device, gpu = init_device()

    # 5. Initialize model
    model = init_model(args['prednet']['channels'],
                       args['prednet']['kernel'],
                       args['prednet']['padding'],
                       args['prednet']['stride'],
                       args['prednet']['dropout'],
                       args['prednet']['peephole'],
                       args['prednet']['pixel_max'],
                       console.get_mode(),
                       gpu).to(device)

    debug = args['debug']
    if debug:
        print_model(model)

    # 6. Initialize dataset
    dataset = init_dataset(console.get_dataset(), args['data_path'],
                           console.get_testing())
    
    # 7 Get dataloader from dataset
    dataloader = get_dataloader(dataset, console.get_batch(),
                                not console.get_testing(), True) # todo: change static True

    # 8. Initialize optimizer
    optim = init_optimizer(console.get_optimizer(), model, console.get_lr())

    # 9. Initialize lr scheduler
    schedule = init_scheduler(optim, console.get_epoch() // 2)

    # 10. Load pre-trained model
    depoch, diteration = 0, 0
    if console.get_load():
        model, depoch, diteration, optim, dloss = load_model(model, optim,
                                                             console.get_dataset(),
                                                             args['mdl_path'])

    # 11. Initialize logger
    tensorlog = TensorLog(args['log_path'], model.name, console.get_testing(),
                          debug)
    tensorlog.open_writer()

    # 12. Train/Test the model
    compute(console.get_testing(), model, optim, schedule, console.get_loss(),
            dataloader, device, tensorlog, console.get_epoch(),
            depoch, diteration, console.get_save(),
            console.get_validate(), console.get_normalize(),
            console.get_binarize(), debug)

    # 13. Close logger
    tensorlog.close_writer()

    # 14. Save model
    save_model(model, optim, console.get_dataset(), args['mdl_path'])
    

if __name__ == '__main__':
    print('Start')
    main()
    print('Done')