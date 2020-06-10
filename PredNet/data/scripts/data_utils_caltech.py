#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@date: 04.06.2020
@author: Sören S. Dittrich
@version: 0.0.1
@description: Read and process Caltech sequence files
"""

import re
import os
import glob
import cv2 as cv
import numpy as np
import hickle as hkl
from PIL import Image
from itertools import chain


PATH='.'
SIZE = (128, 160)
FPS = 10


def process_image(image, size):
    """
    Process the image (Crop, Resize, ...)
    Mostly copied from prednet implementation, to be consistent in cropping.
    
    image := input image
    size := output size
    """
    target_ds = float(size[0])/np.shape(image)[0]
    image = np.array(Image.fromarray(image).resize((int(np.round(target_ds \
                     * np.shape(image)[1])), size[0])))
    d = int((np.shape(image)[1] - size[1]) / 2)
    image = image[:, d:d+size[1]]
    return image


def read_sequence(file, fps, size):
    """
    Read every sequence as is
    
    file := sequence file
    fps := fps to use
    """
    i = FPS
    x = list()
    cap = cv.VideoCapture(file)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if i == FPS:
            x.append(process_image(frame, size))
            i = 0
        i++
        

    return x


def main(path):
    """
    Main file
    
    path := 
    """
    X_train = list()
    train_sources = list()
    X_test = list()
    test_sources = list()

    # training set
    training = [f for f in os.listdir(path) if re.search(r'set0[0-5]', f)]
    for folder in training:
        for file in sorted(glob.glob('{}/*.seq'.format(folder))):
            out = np.array(read_sequence(file, FPS, SIZE))
            X_train.append(out)
            train_sources.append([file] * len(out))
    # test set
    test = [f for f in os.listdir(path) if re.search(r'set(0[6-9]|10)', f)]
    for folder in test:
        for file in sorted(glob.glob('{}/*.seq'.format(folder))):
            out = np.array(read_sequence(file, FPS, SIZE))
            X_test.append(out)
            test_sources.append([file] * len(out))

    hkl.dump(list(chain(*X_train)), os.path.join(path,
             'caltech_train.hkl'))
    hkl.dump(list(chain(*train_sources)), os.path.join(path,
             'caltech_train_sources.hkl'))
    hkl.dump(list(chain(*X_test)), os.path.join(path,
             'caltech_test.hkl'))
    hkl.dump(list(chain(*train_sources)), os.path.join(path,
             'caltech_test_sources.hkl'))


if __name__ == '__main__':
    main(PATH)
