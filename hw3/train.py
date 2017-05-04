## library
import sys
import numpy as np
import pandas as pd
import time
from scipy import ndimage
#import matplotlib.pyplot as plt
## keras
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

modelfile = './model.h5'

#####################

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
	
def testDataLoad(file):
	dataX = pd.read_csv(file)
	image = np.asarray(dataX.loc[:,'feature'].str.split(expand=True), dtype=np.float32)
	image = image.reshape(image.shape[0], in_row, in_col, in_channel)
	
	flipImag = image[:,:,::-1,:]/255
	xtest = np.concatenate((image/255, flipImag))
	
	return xtest
	
def gen_model():
	img_input = layers.Input(shape=(in_row,in_col,in_channel))
	# Block 1
	x = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)
	x = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv3')(x)
	x = MaxPooling2D((2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)
	x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv2')(x)
	x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv3')(x)
	x = MaxPooling2D((2, 2), name='block2_pool')(x)
	
	# Block 3
	x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv1')(x)
	x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv2')(x)
	x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), name='block3_pool')(x)
	
	# Block 4
	x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), name='block4_pool')(x)
	
	# Flatten
	fc = Flatten()(x)
	
	# Fully connected layers
	fc = Dense(2048, activation='relu')(fc)
	fc = Dropout(0.2)(fc)
	
	fc = Dense(1024, activation='relu')(fc)
	fc = Dropout(0.2)(fc)
	
	fc = Dense(512, activation='relu')(fc)
	fc = Dropout(0.2)(fc)

	fc = Dense(256, activation='relu')(fc)
	fc = Dropout(0.2)(fc)

	fc = Dense(128, activation='relu')(fc)
	fc = Dropout(0.2)(fc)

	pred = Dense(out_dim, activation='softmax')(fc)
	
	
	model = Model(inputs=img_input, outputs=pred)
	opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	model.compile(loss='categorical_crossentropy', 
				  optimizer=opt, 
				  metrics=['accuracy'])
	
	return model
	
def testOut(file, y):
	f = open(file, 'w')
	f.write("id,label\n")
	for i in range (7178):
		f.write(str(i) + "," + str(np.sum(y[i])) + "\n")
	f.close()
	
def label_test(xtest, xtrain, ytrain, model, threshold=0.9):
	# Predict unlabel data
	y_new_label = model.predict(xtest, batch_size=batch)
	ytest = np.argmax(y_new_label, axis = 1)
	prob = np.max(y_new_label, axis = 1)
	top_k = np.sum(prob > threshold)
	
	# Get top k index
	if(top_k != 0.0):
		idx = prob.argsort()
		xtrain_new = xtest[np.array(idx[-1*top_k:])]
		ytrain_new = y_new_label[np.array(idx[-1*top_k:])]
		xtest_new = xtest[np.array(idx[0:len(xtest)-top_k])]
	
		X_train = np.concatenate((xtrain, xtrain_new))
		y_train = np.concatenate((ytrain, ytrain_new))
	else:
		xtest_new = xtest
		X_train = xtrain
		y_train = ytrain
	
	return xtest_new, X_train, y_train
	
if __name__ == '__main__':
	## time calculation: starting point
	start_time = time.time()

	## parse data
	print("=================================")
	print("==== parsing train/test data ====")
	print("=================================")
	xtrain, ytrain = trainDataLoad(sys.argv[1])
	#xtest = testDataLoad(sys.argv[2])
	
	## training
	model = gen_model()
	model.summary()
	
	#print('test_size = ', xtest.shape)
	print('train_size = ', xtrain.shape)
	hist = model.fit(xtrain, ytrain, batch_size=batch, epochs=epoch, shuffle=True)
	
	## training process
	#plt.plot(hist.history['acc'])
	#plt.plot(hist.history['val_acc'])
	#plt.title('Model Accuracy')
	#plt.ylabel('Accurary')
	#plt.xlabel('Epochs')
	#plt.legend(['Train', 'Test'])
	#plt.draw()
	#plt.savefig('Training_Processing.png')
	
	
	model.save(modelfile)
	#plot_model(model, to_file='model_hw3.png')
	## predicting
	#ytest = model.predict(xtest, batch_size=batch)
	#ytest = np.argmax(ytest, axis=1)
	#testOut(sys.argv[3], ytest)
	
	## time calculation: ending point
	print("--- %s seconds ---" % (time.time() - start_time))
	
	


