#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras import layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils import plot_model

in_row = 48
in_col = 48
in_channel = 1
out_dim = 7

batch = 100
epoch = 30
self_iter = 10


def trainDataLoad(file):
	dataX = pd.read_csv(file)
	label = np.asarray(dataX.loc[:,'label'], dtype=np.float32)
	image = np.asarray(dataX.loc[:,'feature'].str.split(expand=True), dtype=np.float32)
	image = image.reshape(image.shape[0], in_row, in_col, in_channel)
	
	flipImag = image[:,:,::-1,:]/255
	
	ytrain = np_utils.to_categorical(label, out_dim)
	
	xtrain = np.concatenate((image/255, flipImag))
	ytrain = np.concatenate((ytrain, ytrain))
	
	return xtrain, ytrain

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    step = 1
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * step
    return input_image_data.reshape(48,48).tolist()

def visfilter():
    NUM_STEPS = 20
    RECORD_FREQ = 20
    num_step = 20

    emotion_classifier = load_model('model_for_hw.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ['block2_conv1']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    nb_filter = 64
    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        fig = plt.figure(figsize=(14, 6))
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])

            ###
            # "You need to implement it."
            filter_imgs = grad_ascent(num_step, input_img_data, iterate)
            ###

            filter_imgs = np.asarray(filter_imgs)
            ax = fig.add_subplot(nb_filter/16, 16, filter_idx+1)
            plt.imshow(filter_imgs, cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('Filter '+str(filter_idx+1))
            plt.tight_layout()
        fig.suptitle('Filters of layer conv 1 @ block 1')
        plt.show()

def main():
    emotion_classifier = load_model('model_for_hw.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['block1_conv1', 'block2_conv1']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    xtrain, ytrain = trainDataLoad('train.csv')
    private_pixels = xtrain
    
    choose_id = 0
    photo = private_pixels[choose_id].reshape(1, 48, 48, 1)
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('Filter '+str(i+1))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(name_ls[cnt], choose_id))
        plt.show()


if __name__ == "__main__":
    #visfilter()
    main()

