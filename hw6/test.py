import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential, Model, load_model
from keras import layers
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import concatenate, dot, Dot, Add
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils import plot_model
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers.wrappers import Bidirectional
import keras.backend as K

TEST_PATH = sys.argv[1] + 'test.csv'
MODEL_FILE = 'model_v1.h5'
OUT_FILE = sys.argv[2]
RNG_SEED = 1446557
K_FACTORS = 120


def Read(test):
	data_test = pd.read_csv(test)
	test_users = data_test['UserID'].values - 1
	test_movies = data_test['MovieID'].values - 1
	return (test_users, test_movies)

	
def main():
	(test_users, test_movies) = Read(TEST_PATH)
	print("Reading file successfully")	
	model = load_model(MODEL_FILE)
	test_rating = model.predict([test_users, test_movies])
	print("Testing successfully")
	
	f = open(OUT_FILE, 'w')
	f.write("TestDataID,Rating\n")
	for i in range(test_rating.shape[0]):
		f.write("%d,%f\n" %(i+1, test_rating[i]))
	f.close()
	print("Output file successfully")
	
	
if __name__ == '__main__':
	main()



