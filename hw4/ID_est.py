from sklearn.decomposition import PCA
import numpy as np
import time
import pandas as pd
import sys


def ID_est(setData):
	mean = np.mean(setData.T, axis=0)
	data = setData.T - np.tile(mean.reshape(1, mean.shape[0]), (setData.shape[1], 1))
	U, s, V = np.linalg.svd(data, full_matrices=False)
	err = []
	
	for i in range(60):
		print('processing with dimension %d' %(i))
		dim = i + 1                                                # choosing approximate dimension
		
		S = np.zeros((dim, dim))                                   # diagonal matrix (eigenvalues)
		S[:,:] = np.diag(s[:dim])
		
		reconstruct = np.dot(U[:,:dim], np.dot(S, V[:dim, :]))     # reconstruction data
	
		rmse = np.sqrt( np.mean( np.square(np.abs(data - reconstruct)), axis=0) )
		rmse = np.mean(rmse)
		
		err = np.append(err, rmse)
		if(rmse < 7.0):
			break
	
	return err
	
def readRMSE():
	data = pd.read_csv('RMSE.csv', header=None)
	data = np.asarray(data)
	id = np.zeros((200, ))
	
	for i in range(200):
		id[i] = np.sum(data[i,:] >= 7.0) 
		print(id[i])
	
	return id
	
	
if __name__ == '__main__':
	start_time = time.time()
	
	data = np.load(sys.argv[1])
	err = np.zeros((200, 60))
	
	f = open(sys.argv[2], 'w')
	f.write('SetId,LogDim\n')
	for i in range(200):
		print('======================')
		print('processing set %d' %(i))
		print('======================')
		setData = data[str(i)]
		rmse = ID_est(setData)
		
		if ((rmse.shape[0]) == 0 ):
			f.write('%d,0\n' %(i))
		else:
			f.write('%d,%f\n' %(i, np.log(rmse.shape[0])))
		
		#err[i,:] = rmse
	f.close()
	
	#f = open('RMSE.csv', 'w')
	#for i in range(200):
	#	for j in range(60):
	#		f.write('%f' %(err[i, j]))
	#		if (j != 59):
	#			f.write(',')
	#			
	#	f.write('\n')
	#f.close()
		
	#id = readRMSE()
	#f = open('test.csv', 'w')
	#f.write('SetId,LogDim\n')
	#for i in range(200):
	#	if (id[i] != 0):
	#		f.write('%d,%f\n' %(i, np.log(id[i])))
	#	else:
	#		f.write('%d,0\n' %(i))
	#f.close()
	
	print("--- %s seconds ---" % (time.time() - start_time))

