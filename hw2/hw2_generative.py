import sys
import numpy as np
import pandas as pd
import time

def trainDataLoad(fx, fy):
	dataX = pd.read_csv(fx)
	dataX = np.matrix(dataX, dtype=float)
	
	dataY = pd.read_csv(fy, header=None)
	dataY = np.matrix(dataY, dtype=float)
	
	num = np.sum(dataY, dtype=int)
	
	dataX0 = np.matrix(np.zeros([dataY.shape[0]-num, dataX.shape[1]]))
	dataX1 = np.matrix(np.zeros([num, dataX.shape[1]]))
	
	i0 = 0
	i1 = 0
	for i in range(dataY.shape[0]):
		if (dataY[i,:] == 0):
			dataX0[i0,:] = dataX[i,:]
			i0 = i0 + 1
		else:
			dataX1[i1,:] = dataX[i,:]
			i1 = i1 + 1
	
	return dataX0, dataX1, dataY

def testDataLoad(fx):
	dataX = pd.read_csv(fx)
	dataX = np.matrix(dataX, dtype=float)
	
	return dataX




if __name__ == '__main__':
	
	## time calculation: starting point
	start_time = time.time()
	
	## import order: 
	## 1: train.csv, 2: test.csv, 3: X_train, 4: Y_train, 5: X_test, 6: output file
	
	trainX0, trainX1, trainY = trainDataLoad(sys.argv[3], sys.argv[4])
	testX = testDataLoad(sys.argv[5])
	
	pc0 = np.float(trainX0.shape[0])/np.float(trainY.shape[0])
	pc1 = np.float(trainX1.shape[0])/np.float(trainY.shape[0])
	
	covC0 = np.matrix(np.zeros([106, 106]))
	covC1 = np.matrix(np.zeros([106, 106]))
	## this is for class 0
	meanC0 = np.mean(trainX0, axis=0)	
	for i in range(trainX0.shape[0]):
		covC0 = covC0 + (trainX0[i,:] - meanC0).T * (trainX0[i,:] - meanC0) / np.float(trainX0.shape[0])	
	## this is for class 1
	meanC1 = np.mean(trainX1, axis=0)
	for i in range(trainX1.shape[0]):
		covC1 = covC1 + (trainX1[i,:] - meanC1).T * (trainX1[i,:] - meanC1) / np.float(trainX1.shape[0])

	## mix
	cov = (pc0 * covC0) + (pc1 * covC1)
	covI = np.linalg.inv(cov)
	det = np.linalg.det(cov)
	
	
	f = open(sys.argv[6], 'w')
	f.write("id,label\n")
	for i in range(testX.shape[0]):
		pxC0 = np.exp(-0.5 * (testX[i,:] - meanC0) * covI * (testX[i,:] - meanC0).T)
		pxC1 = np.exp(-0.5 * (testX[i,:] - meanC1) * covI * (testX[i,:] - meanC1).T)
		
		pC0x = pxC0 * pc0 / (pxC0 * pc0 + pxC1 * pc1)
		if (pC0x > 0.5):
			f.write(str(i+1) + ",0\n")
		else:
			f.write(str(i+1) + ",1\n")
		
	f.close()
	
	## time calculation: ending point
	print("--- %s seconds ---" % (time.time() - start_time))
