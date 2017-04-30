## library
import sys
import numpy as np
import pandas as pd
import time
from scipy import ndimage
#import matplotlib.pyplot as plt
## keras
from keras.models import Sequential, Model, load_model
from keras import layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

## global constant ##
in_row = 48
in_col = 48
in_channel = 1
out_dim = 7

batch = 100
epoch = 30
self_iter = 10

def testDataLoad(file):
	dataX = pd.read_csv(file)
	image = np.asarray(dataX.loc[:,'feature'].str.split(expand=True), dtype=np.float32)
	image = image.reshape(image.shape[0], in_row, in_col, in_channel)
	
	flipImag = image[:,:,::-1,:]/255
	xtest = np.concatenate((image/255, flipImag))
	
	return xtest
	
def testOut(file, y):
	f = open(file, 'w')
	f.write("id,label\n")
	for i in range (7178):
		f.write(str(i) + "," + str(np.sum(y[i])) + "\n")
	f.close()
	
if __name__ == '__main__':
	## time calculation: starting point
	start_time = time.time()

	## parse data
	print("=================================")
	print("===== parsing testing data ======")
	print("=================================")
	xtest = testDataLoad(sys.argv[1])
	
	
	print("=================================")
	print("====== loading model v4~v8 ======")
	print("=================================")
	model1 = load_model('model_selftraining_10_v7.h5')
	model2 = load_model('model_selftraining_10_v5.h5')
	model3 = load_model('model_selftraining_10_v8.h5')
	model4 = load_model('model_selftraining_10_v6.h5')
	model5 = load_model('model_selftraining_10_v4.h5')

	## predicting
	print("=================================")
	print("========== prediecting ==========")
	print("=================================")
	ytest1 = model1.predict(xtest, batch_size=batch)
	ytest2 = model2.predict(xtest, batch_size=batch)
	ytest3 = model3.predict(xtest, batch_size=batch)
	ytest4 = model4.predict(xtest, batch_size=batch)
	ytest5 = model5.predict(xtest, batch_size=batch)
	
	ytest = ytest1 + ytest2 + ytest3 + ytest4 + ytest5
	
	ytest = np.argmax(ytest, axis=1)
			
	testOut(sys.argv[2], ytest)
	
	## time calculation: ending point
	print("--- %s seconds ---" % (time.time() - start_time))

