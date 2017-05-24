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

DATA1 = 'rnn_v6.pickle'
DATA2 = 'rnn_v4.pickle'
DATA3 = 'rnn_v1.pickle'
DATA4 = 'rnn_v5.pickle'
DATA5 = 'rnn_v3.pickle'
#DATA6 = 'rnn_v9.pickle'
MODEL1 = './model/rnn_v6.hdf5'
MODEL2 = './model/rnn_v4.hdf5'
MODEL3 = './model/rnn_v1.hdf5'
MODEL4 = './model/rnn_v5.hdf5'
MODEL5 = './model/rnn_v3.hdf5'
#MODEL6 = './model/rnn_v9.hdf5'
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

def result_write(y_test, thresh):
	f = open(OUT, 'w')
	f.write("\"id\",\"tags\"\n")
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
	print("Loading Testing Data")
	x_test1 = pickle.load(open(DATA1, 'rb'))
	x_test2 = pickle.load(open(DATA2, 'rb'))
	x_test3 = pickle.load(open(DATA3, 'rb'))
	x_test4 = pickle.load(open(DATA4, 'rb'))
	x_test5 = pickle.load(open(DATA5, 'rb'))
	#x_test6 = pickle.load(open(DATA6, 'rb'))
	
	print("Loading Models")
	model1 = load_model(MODEL1, custom_objects={'f1_score':f1_score})
	model2 = load_model(MODEL2, custom_objects={'f1_score':f1_score})
	model3 = load_model(MODEL3, custom_objects={'f1_score':f1_score})
	model4 = load_model(MODEL4, custom_objects={'f1_score':f1_score})
	model5 = load_model(MODEL5, custom_objects={'f1_score':f1_score})
	#model6 = load_model(MODEL6, custom_objects={'f1_score':f1_score})
	
	print("Predicting")
	y_test1 = model1.predict(x_test1)
	y_test2 = model2.predict(x_test2)
	y_test3 = model3.predict(x_test3)
	y_test4 = model4.predict(x_test4)
	y_test5 = model5.predict(x_test5)
	#y_test6 = model6.predict(x_test5)
	#y_test = y_test1 + y_test2 + y_test3 + y_test4 + y_test5 + y_test6
	
	
	thresh = 0.4
	y_pred1 = (y_test1 > thresh).astype('int')
	y_pred2 = (y_test2 > thresh).astype('int')
	y_pred3 = (y_test3 > thresh).astype('int')
	y_pred4 = (y_test4 > thresh).astype('int')
	y_pred5 = (y_test5 > thresh).astype('int')
	#y_pred6 = (y_test6 > thresh).astype('int')
	
	y = y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5# + y_pred6
	result_write(y, 2.0)
		
if __name__ == '__main__':
	main()
	

	
