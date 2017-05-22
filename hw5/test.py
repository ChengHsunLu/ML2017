import sys
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential, Model, load_model
from keras import layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils import plot_model
import keras.backend as K

DATA_PATH = 'rnn_v1.pickle'
MODEL = './model/rnn_v1.hdf5'
with open('TAGS.pickle', 'rb') as handle:
    all_tags = pickle.load(handle)
TEST_PATH = sys.argv[1]
OUT = sys.argv[2]

	
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def result_write(y_test):
	f = open(OUT, 'w')
	f.write("\"id\",\"tags\"\n")
	thresh = 0.4
	y_pred_thresh = (y_test > thresh).astype('int')
	for i in range(y_test.shape[0]):
		y = y_pred_thresh[i]
		tag = []
		for n in range(38):
			if (y[n] == 1):
				tag.extend([all_tags[n]])
		
		tag = ' '.join(tag)
		#if(np.all(y) == 0):
		#	idx = np.argmax(y_test[i])
		#	tag = all_tags[idx]
		
		f.write("\"%d\",\"%s\"\n" %(i, tag))
	
	f.close()

def main():
	with open(DATA_PATH, 'rb') as handle:
		x_test = pickle.load(handle)
	model = load_model(MODEL, custom_objects={'f1_score':f1_score})
	y_test = model.predict(x_test)
	result_write(y_test)
		
if __name__ == '__main__':
	main()
	

	
