import os
import pandas as pd
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras import layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils import plot_model
## global constant ##
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


def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    args = parser.parse_args()
    
    emotion_classifier = load_model('model_for_hw.h5')

    xtrain, ytrain = trainDataLoad('train.csv')
    private_pixels = xtrain
    input_img = emotion_classifier.input
    img_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for idx in img_ids:
        #private_pixels = xtrain[idx]
        val_proba = emotion_classifier.predict(private_pixels[idx].reshape(1,48,48,1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        heatmap = fn([private_pixels[idx].reshape(1,48,48,1), 0])
        heatmap = np.asarray(heatmap).reshape(48,48)


        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        see[np.where(np.absolute(heatmap) <= thres)] = np.mean(see)


        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('heatmap'+str(idx)+'.png', dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('result'+str(idx)+'.png', dpi=100)

if __name__ == "__main__":
    main()

