

import sys
import numpy as np
import pandas as pd
import time

def trainDataParser(filename):
	rawData = pd.read_csv(filename, header=None, encoding='big5')
	rawData = np.array(rawData)
	# find the location where store AMB_TEMP (for date/hour start points)
	"""	loc[0][]: raw, loc[1][]: col """
	startLoc = np.array(np.where(rawData=='AMB_TEMP'))
	# replace NR into 0
	rawData[rawData == 'NR'] = 0
	
	# 18 measured sensor name
	label = rawData[startLoc[0,0]: startLoc[0,0]+18, startLoc[1,0]]
	# combined the 18 sensor data into 18 rows (col means time)
	for n in range(0, label.size, 1):
		sensorData = rawData[ rawData[:,startLoc[1,0]]==label[n] ]
		sensorData = sensorData[:, startLoc[1,0]+1:]
		sensorData = sensorData.astype(float)
		sensorData = sensorData.ravel()
		if n == 0:
			data = sensorData
		else:
			data = np.vstack((data, sensorData))
	
	# consecutive 9 hours forms a feature set (18x9 features in each)
	# use the above feature set to predict the following PM2.5
	# divide each month
	for mon in range(12):
		for n in range(471):
			feature = data[:, 480*mon+n:480*mon+n+9]
			feature = feature.ravel()
			#feature = np.append(feature, feature**2)
			#feature = np.append(feature, data[15, 480*mon+n:480*mon+n+9])
			if (mon==0 and n==0):
				trainDataX = feature
				trainDataY = data[9, 480*mon+n+9]
			else:
				trainDataX = np.vstack((trainDataX, feature))
				trainDataY = np.append(trainDataY, data[9, 480*mon+n+9])
	
	trainDataX = np.insert(trainDataX, 0, 1, axis=1) ## insert 1 in col=0
	trainDataX = np.matrix(trainDataX)
	opt = np.dot(np.dot(np.dot(trainDataX.T, trainDataX).I, trainDataX.T), trainDataY)
	
	return trainDataX, trainDataY, opt
	
def testDataParser(filename):
	rawData = pd.read_csv(filename, header=None)
	rawData = np.array(rawData)
	startLoc = np.array(np.where(rawData=='AMB_TEMP'))
	rawData[rawData == 'NR'] = 0
	
	for n in range(startLoc.shape[1]):
		feature = rawData[startLoc[0,n]:startLoc[0,n]+18, startLoc[1,n]+1:]
		feature = feature.astype(float)
		feature = feature.ravel()
		#feature = np.append(feature, feature**2)
		#feature = np.append(feature, rawData[startLoc[0,n]+15, startLoc[1,n]+1:])
		if n==0 :
			testDataX = feature
		else :
			testDataX = np.vstack((testDataX, feature))
	
	testDataX = np.insert(testDataX, 0, 1, axis=1)
	testDataX = testDataX.astype(float)
	testDataX = np.matrix(testDataX)
	
	return testDataX
	
def resultWrite(filename, testY):
	f = open(filename, 'w')
	f.write("id,value\n")
	for i in range (0,testY.shape[1],1):
		f.write("id_" + str(i) + "," + str(np.sum(testY[:,i])) + "\n")
	f.close()
	
def weightLoad(filename):
	rawData = pd.read_csv(filename, header=None)
	rawData = np.array(rawData)
	rawData = rawData.astype(float)
	return rawData
	
class LinearRegression:
	def __init__(self, dataDim, optParameter=None):
		if (np.all(optParameter) != None):
			w = optParameter
		else:
			w = np.matrix(0.01*np.random.rand(dataDim))
		
		self.w = w
			
	def train(self, trainDataX, trainDataY, maxEpoch, learningRate=1e-2, lb=0):
		adaGrad = np.zeros(trainDataX.shape[1]) + 1e-12
		trainError = []
		
		## Training ##
		print('---------------------------Start training---------------------------')
		for epoch in range(maxEpoch):
			yPred = np.array(np.dot(trainDataX, self.w.T).T)
			loss = np.sqrt(np.sum(((yPred - trainDataY))**2)/trainDataX.shape[0])
			w_grad = 2*np.dot(trainDataX.T, (yPred - trainDataY).T)
			w_grad[1:] = w_grad[1:] + 2*lb*(self.w[:,1:].T) ## regularization
			adaGrad = adaGrad + np.array(w_grad).T**2 ## adagrad
			self.w = self.w - lr*w_grad.T/np.sqrt(adaGrad)
			trainError = np.append(trainError, loss)
			print("Training..., (%d/%d): Loss = %f" % (epoch, maxEpoch, loss))
		
		return trainError
	
	def test(self, testDataX):
		testY = np.array(np.dot(testDataX, self.w.T).T)
		return testY
	
	def valid(self, validX, validY):
		yPred = np.dot(trainDataX, self.w.T)
		loss = np.sqrt(np.sum(nu.multiply(yPred - validY, yPred - validY))/validX.shape[0])
		
		return loss
	
if __name__ == '__main__':
	start_time = time.time()
	trainDataX, trainDataY, opt = trainDataParser(sys.argv[1])
	testDataX = testDataParser(sys.argv[2])
	
	
	## training parameters
	lr = 1e-1
	lb = 1e-2
	maxEpoch = 500000
	
	w = weightLoad("./w_162.csv")
	
	linearRegression = LinearRegression(trainDataX.shape[1], w.T)
	#trainError = linearRegression.train(trainDataX, trainDataY, maxEpoch, lr, lb)
	testY = linearRegression.test(testDataX)
	
	## write the parameters
	#f = open(sys.argv[4], 'w')
	#for i in range (linearRegression.w.shape[1]):
	#	f.write(str(np.sum(linearRegression.w[:,i])) + "\n")
	#f.close()
	
	resultWrite(sys.argv[3], testY)
	print("--- %s seconds ---" % (time.time() - start_time))
	