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
from model.autoenc import AutoENC
from model.spatiotemp import AE_ConvLSTM_flow as SpatioTemp

# datasets
from dataset.MovingMNIST import MovingMNIST
from dataset.Kitti import Kitti
from dataset.Kth import Kth
from dataset.Caltech import Caltech

# helper imports
from helper.arguments import ConsoleArguments
from helper.yaml_parser import yml
from helper.mdl import ModelLoader
from helper.mdl import ModelSaver
from helper.mdl import ModelParameter
from helper.tensorboard_log import TensorLog
from helper.plot_graph import plot_graph
from helper.early_stopping import EarlyStopping

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
    gpu = False
#    device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu = True
        
    return device, gpu


def init_prednet(channels, kernel, padding, stride, dropout,
                 peephole, pixel_max, mode, predrnn, extrapolate, gpu):
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
    predrnn := Use ConvLSTM or PredRNN
    extrapolate := extrapolate t+n images into future
    gpu := model uses gpu
    """   
    model = PredNet(channels, kernel, padding, stride, dropout,
                    peephole, pixel_max, mode, predrnn, extrapolate, gpu)

    return model


def init_convlstm(depth, channel, kernel, padding, dropout,
                  peephole, predrnn, extrapolate, gpu):
    """
    Initialize ConvLSTM with arguments from yml file

    depth := depth of the autoencoder
    channel := list of channel
    kernel := kernel size
    padding := padding size
    dropout := percentage [0,1]
    peephole := LSTM using peephole
    predrnn := Use ConvLSTM or PredRNN
    extrapolate := extrapolate t+n images into future
    gpu := model uses gpu
    """
    model = AutoENC(depth, channel, kernel,
                    padding, predrnn, gpu)

    return model


def init_spatio(encoder, lstm, flow, huberp, grid, decoder,
                peephole, predrnn, extrapolate, gpu):
    """
    """
    model = SpatioTemp(encoder, lstm, flow, huberp, grid, decoder,
                       forecast=extrapolate+1, gpu=gpu, predrnn=predrnn)
    
    return model
    

def print_model(model):
    """
    Printing the models trainable parameter and the state_dict

    model := initialized network model
    """
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[DEBUG] ' + model.name + ' consists of: ' + str(trainable_param) +
      ' trainable parameters.')    
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def init_dataset(dataset, root, testing, seq_len, download=True):
    """
    Initialize available datasets for training and validation or testing

    dataset := name of the dataset <mnist|kitti|kth|caltech>
    testing := return testing set if True otherwise training set
    seq_len := length of required sequence
    download := download dataset
    """
    data = []
    
    if not testing:
        if dataset == 'mnist':
            data.append(MovingMNIST(root, seq_len, testing))
            data.append(MovingMNIST(root, seq_len, testing, True))
        elif dataset == 'kitti':
            data.append(Kitti(root, seq_len, testing))
            # todo
        elif dataset == 'kth':
            data.append(Kth(root, seq_len, testing))
            data.append(Kth(root, seq_len, testing, True))
            # todo
        elif dataset == 'caltech':
            data.append(Caltech(root, seq_len, testing))
            # todo
        else:
            raise IOError('[ERROR] Choose a valid dataset <mnist|kitti|kth|caltech>')
    else:
        if dataset == 'mnist':
            data.append(MovingMNIST(root, seq_len, testing))
        elif dataset == 'kitti':
            data.append(Kitti(root, seq_len, testing))
        elif dataset == 'kth':
            data.append(Kth(root, seq_len, testing))
        elif dataset == 'caltech':
            data.append(Caltech(root, seq_len, testing))
        else:
            raise IOError('[ERROR] Choose a valid dataset <mnist|kitti|kth|caltech>')

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
                                                          drop_last=drop,
                                                          num_workers=4))
        else:
            dataloader.append(torch.utils.data.DataLoader(dataset=dataset[i],
                                                          batch_size=batch,
                                                          shuffle=not shuffle,
                                                          drop_last=drop,
                                                          num_workers=4))

    return dataloader


def init_optimizer(optim, model, lr, decay=0.9):
    """
    Initialize the optimizer

    optim := name of optimizer <adam|rmsprop>
    model := initialized network model
    lr := learning rate
    decay := weight decay for RMSProp
    """
    if optim == 'adam':
        optimizer = Optim.Adam(model.parameters(), lr)
    elif optim == 'rmsprop':
        optimizer = Optim.RMSprop(model.parameters(), lr, weight_decay=decay)    
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
    dataset := name of used dataset <mnist|kitti|kth|caltech>
    path := path from where to load the model
    debug := print debug output -> False
    """
    loader = ModelLoader(dataset, device, path, model.name, debug)
    params = loader.load()

    model.load_state_dict(params.mdl_state)
    optimizer.load_state_dict(params.optim_state)

    return model, params.epoch, params.iteration, optimizer, params.loss


def compute(name, testing, model, optimizer, scheduler, loss, early, dataloader,
            device, logger, epoch, iteration, depoch, diteration, save, validate,
            normalize, binarize, time_weight, layer_weight, debug=False):
    """
    Compute test or training, given the flag testing

    testing := if true perform testing, otherwise training
    name := name of the model
    model := initialized network model
    optimizer := initialized optimizer
    scheduler := initialized scheduler
    loss := loss function to use
    early := initialized early stopping
    dataloader := initialized dataloader
    device := GPU or CPU
    logger := tensorboard logger
    epoch := epochs to perform
    iteration := number of iterations to perform per epoch
    depoch := already performed epochs (Only pre-trained model)
    diteration := already performed iterations in peoch (Only pre-trained model)
    save := True if model should be saved, False otherwise
    validate := validate every n epochs
    normalize := normalize the image input
    binarize := binarize the image input
    time_weight := time weight parameters for error loss
    layer_weight := layer weight parameters for error loss
    debug := debug value
    """
    if not testing:
        train(name, model, optimizer, scheduler, loss, early, dataloader, device,
              logger, epoch, iteration, save, validate, depoch, diteration,
              normalize, binarize, time_weight, layer_weight, debug)
    else:
        with torch.no_grad():
            test(name, model, iteration, loss, dataloader[0], logger, device,
                 normalize, binarize)


def save_model(model, optimizer, dataset, path, debug=False):
    """
    Save the model at the end

    model := initialized network model
    optimizer := initialized optimizer
    dataset := name of used dataset <mnist|kitti|kth|caltech>
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


def create_error_weights(time_weights, layer_weights, seq_len, layer, gpu):
    """
    Create the time and layer weights for the loss module

    time_weights := both values from the yml file
    layer_weights := both values from the yml file
    seq_len := length of input sequence
    layer := amount of layer used in the network
    gpu := GPU or CPU
    """
    l1 = time_weights[0]
    if gpu:
        layer_weights = torch.FloatTensor([[time_weights[1] for i in range(seq_len - 1)]]).cuda()
        layer_weights = torch.cat((torch.FloatTensor([[l1]]).cuda(), layer_weights), 1).T
        time_weights = (1. / (layer - 1) * torch.ones(layer, 1)).cuda()
    else:
        layer_weights = torch.FloatTensor([[time_weights[1] for i in range(seq_len - 1)]])
        layer_weights = torch.cat((torch.FloatTensor([[l1]]), layer_weights), 1).T
        time_weights = 1. / (layer - 1) * torch.ones(layer, 1)
    time_weights[0] = 0

    return time_weights, layer_weights


def main():
    """
    Main file: Where the magic happens ;)
    """
    # 1. Initialize console arguments
    console = init_arguments()
    name = console.get_model()

    # 2. Initialize yml parameter
    args = init_yml(console.get_parameter())

    # 3. Check if yml parameter and console arguments are given (Console args are favored)
    # todo: implement

    # 4. Initialize device (GPU or CPU)
    device, gpu = init_device()

    if args['debug']:
        print('Model will run on: ' + str(device))

    # 5. Initialize model
    if name == 'prednet':   
        model = init_prednet(args[name]['channels'],
                             args[name]['kernel'],
                             args[name]['padding'],
                             args[name]['stride'],
                             args[name]['dropout'],
                             args[name]['peephole'],
                             args[name]['pixel_max'],
                             console.get_mode(),
                             console.get_predrnn(),
                             console.get_extrapolate(),
                             gpu).to(device)
    elif name == 'convlstm':
        model = init_convlstm(args[name]['depth'],
                              args[name]['channel'],
                              args[name]['kernel'],
                              args[name]['padding'],
                              args[name]['dropout'],
                              args[name]['peephole'],
                              console.get_predrnn(),
                              console.get_extrapolate(),
                              gpu).to(device)
    else:
        model = init_spatio(args[name]['encoder'],
                            args[name]['lstm'],
                            args[name]['flow'],
                            args[name]['huber'],
                            args[name]['grid'],
                            args[name]['decoder'],
                            args[name]['lstm']['peephole'],
                            console.get_predrnn(),
                            console.get_extrapolate(),
                            gpu).to(device)

    debug = args['debug']
    if debug:
        print_model(model)

    if console.get_plot():
        plot_graph(model, args[name]['size'], args['plot_path'], gpu)

    # 6. Initialize dataset
    dataset = init_dataset(console.get_dataset(), args['data_path'],
                           console.get_testing(),
                           console.get_sequence())

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
        # To recreate the natural image test from the paper (Training on Kitti,
        # Testing on Caltech).
        dset = console.get_dataset()
        if console.get_testing() and dset == 'caltech':
            dset = 'kitti'
        model, depoch, diteration, optim, dloss = load_model(model, optim, device,
                                                             dset,
                                                             args['mdl_path'])

    # 11. Initialize logger
    tensorlog = TensorLog(args['log_path'], model.name, console.get_testing(),
                          debug)
    tensorlog.open_writer()

    # 12. Build error weights, if training
    time_weight, layer_weight = None, None
    if not console.get_testing() and name == 'prednet':
        time_weight, layer_weight = create_error_weights(args['prednet']['seq_weight'],
                                                         args['prednet']['layer_weight'],
                                                         console.get_sequence(),
                                                         len(args['prednet']['channels'][:-1]),
                                                         gpu)

    # 13. Initialize early stopping if training
    early = None
    if not console.get_testing():
        early = EarlyStopping(patience=int(console.get_patience()))

    # 14. Train/Test the model
    compute(name, console.get_testing(), model, optim, schedule,
            console.get_loss(), early, dataloader, device, tensorlog,
            console.get_epoch(), console.get_iteration(), depoch,
            diteration, console.get_save(), console.get_validate(),
            console.get_normalize(), console.get_binarize(),
            time_weight, layer_weight, debug)

    # 15. Close logger
    tensorlog.close_writer()

    # 16. Save model
    if console.get_save():
        save_model(model, optim, console.get_dataset(), args['mdl_path'])


if __name__ == '__main__':
    print('Start')
    main()
    print('Done')
