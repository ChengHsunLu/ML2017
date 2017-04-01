import sys
import numpy as np
import pandas as pd
import time

def trainDataLoad(fx, fy):
	dataX = pd.read_csv(fx)
	dataX = np.matrix(dataX, dtype=float)
	
	dataY = pd.read_csv(fy, header=None)
	dataY = np.matrix(dataY, dtype=float)
	
	Y =  np.vstack((1 - dataY.T, dataY.T))
	
	return dataX, Y

def testDataLoad(fx):
	dataX = pd.read_csv(fx)
	dataX = np.matrix(dataX, dtype=float)
	
	return dataX

def modelLoad(filename):
	rawData = pd.read_csv(filename, header=None)
	rawData = np.array(rawData)
	rawData = rawData.astype(float)
	return np.matrix(rawData)
	
	
class fcLayer:
	## fully connected layer with ReLU
	def __init__(self, n_in, n_out, W=None, b=None):
		"""
			input: symbolic variable
			n: dimension of output
		"""
		if W is None:
			W = np.matrix(0.01*np.random.randn(n_in, n_out), dtype=float)
		if b is None:
			b = np.matrix(0.01*np.random.rand(n_out, 1), dtype=float)

		self.w = W
		self.b = b;

	def forward(self, input, actfor):
		self.innerProduct = self.w.T * input + self.b
		self.out = actfor(self.innerProduct)
	
	def backward(self, error, input, actback):
		## error signal
		self.actError = actback(error, self.innerProduct)
		self.err = self.w * self.actError
		## dE/dW
		self.w_grad = input * self.actError.T
		## dE/db
		self.b_grad = np.sum(self.actError, axis=1);
	
def softmax_for(x):
	n = np.exp(x)
	d = np.sum(np.exp(x), axis=0)
	d = np.tile(d, (x.shape[0], 1))
	#for i in range(x.shape[1]):
	#	n[:,i] = np.divide(n[:,i], d[:,i])
	return np.divide(n, d)
		
def softmax_back(error, x):
	n = np.exp(x)
	d = np.sum(np.exp(x), axis=0)
	d = np.tile(d, (x.shape[0], 1))
	return np.divide(n, d) - error

		
class logistic:
	def __init__(self, n_in, n_out):
		model = modelLoad('my_model.csv')
		self.f0 = fcLayer(n_in, n_out, model[:111,:], model[111,:].T)
		#self.f0 = fcLayer(n_in, n_out)
		
	def train(self, trainX, trainY, maxEpoch, learningRate, lb):
		epoch = 0
		lr = learningRate;
		ada_w = np.zeros(self.f0.w.shape) + 1e-6
		ada_b = np.zeros(self.f0.b.shape) + 1e-6
		alpha = 0.3
		
		print "============ start training ============"
		for epoch in range(maxEpoch):
			## forwardpass
			self.f0.forward(trainX, softmax_for)
			
			if (np.max(self.f0.out) != np.max(self.f0.out)):
				break

			loss = -np.sum(np.multiply(trainY, np.log(self.f0.out)) )
			
			## backwardpass
			self.f0.backward(trainY, trainX, softmax_back)

			## RMSprop
			ada_w = alpha*ada_w + (1-alpha)*np.multiply(self.f0.w_grad, self.f0.w_grad)
			ada_b = alpha*ada_b + (1-alpha)*np.multiply(self.f0.b_grad, self.f0.b_grad)
			ada_w[ada_w == 0] = 1e-6
			ada_b[ada_b == 0] = 1e-6
			
			## update
			self.f0.w = (1-lr*lb)*self.f0.w - np.divide(lr*self.f0.w_grad, np.sqrt(ada_w))
			self.f0.b = (1-lr*lb)*self.f0.b - np.divide(lr*self.f0.b_grad, np.sqrt(ada_b))
			
			print "==== Epoch = ", epoch, ", with loss = ", loss, " ===="
		
		if (epoch != (maxEpoch -1) ):
			print "Error!!! Value is not a number!!!!!! @ epoch = ", epoch
			
	def test(self, testX, file):
		self.f0.forward(testX, softmax_for)
		ans = np.argmax(self.f0.out, axis=0)
		
		f = open(file, 'w')
		f.write("id,label\n")
		for i in range (ans.shape[1]):
			f.write(str(i+1) + "," + str(np.sum(ans[:,i])) + "\n")
		f.close()
		
		#f = open('my_model.csv', 'w')
		#for row in range(self.f0.w.shape[0]):
		#	for col in range(self.f0.w.shape[1]):
		#		f.write(str(np.sum(self.f0.w[row, col])))
		#		if (col == self.f0.w.shape[1]-1):
		#			f.write('\n')
		#		else:
		#			f.write(',')
		#			
		#for row in range(self.f0.b.shape[0]):
		#	f.write(str(np.sum(self.f0.b[row, :])))
		#	if (row == self.f0.b.shape[0]-1):
		#		f.write('\n')
		#	else:
		#		f.write(',')
		#f.close()

if __name__ == '__main__':
	
	## time calculation: starting point
	start_time = time.time()
	
	## import order: 
	## 1: train.csv, 2: test.csv, 3: X_train, 4: Y_train, 5: X_test, 6: output file
	
	trainX, trainY = trainDataLoad(sys.argv[3], sys.argv[4])
	testX = testDataLoad(sys.argv[5])
	
	## feature scaling	
	for i in (0, 1, 3, 4, 5):
		trainX = np.hstack((trainX, np.multiply(trainX[:,i], trainX[:,i])))
		testX = np.hstack((testX, np.multiply(testX[:,i], testX[:,i])))
	
	dataX = np.vstack((trainX, testX))
	dataXmean = np.mean(dataX, axis=0)
	dataXstd = np.std(dataX, axis=0)
	
	for i in (0, 1, 3, 4, 5, 106, 107, 108, 109, 110):
		trainX[:,i] = (trainX[:,i] - dataXmean[:,i])/dataXstd[:,i]
		testX[:, i] = (testX[:,i] - dataXmean[:,i])/dataXstd[:,i]
		

	## model
	mylogistic = logistic(trainX.shape[1], 2)
	#mylogistic.train(trainX.T, trainY, 5000, 0.001, 0.001)
	mylogistic.test(testX.T, sys.argv[6])

	
	## time calculation: ending point
	print("--- %s seconds ---" % (time.time() - start_time))
