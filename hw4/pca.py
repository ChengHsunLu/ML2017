import os, glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt

img_class = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
path = './data/prob1/'
output = 'pcaImage.csv'


def parse():
	f = open(output, 'w')
	
	for c in img_class:
		for idx in range(75):
			if (idx < 10):
				filename = c + '0' + str(idx) + '.bmp'
			else:
				filename = c + str(idx) + '.bmp'
			img = Image.open(os.path.join(path, filename))
			img = np.asarray(img, dtype=np.float)
			img = np.ravel(img)
			
			for i in range(img.shape[0]):
				f.write( str(img[i]) )
				if ( i == img.shape[0] - 1 ):
					f.write('\n')
				else:
					f.write(',')
	
	f.close()
	
def readimg():
	data = pd.read_csv('pcaImage.csv', header=None)
	data = np.matrix(data, dtype=np.float)
	
	for i in range(10):
		if (i == 0):
			img = data[0:10, :]
		else:
			img = np.vstack((img, data[75*i:75*i+10, :]))
	
	return img.T
	
def pca(img, imgMean, num):
	img = img - np.tile(imgMean, (1, img.shape[1]))
	U, s, V = np.linalg.svd(img)
	
	S = np.diag(s[:num])

	reconstruct = np.dot(U[:,:num], np.dot(S, V[:num, :]))
	
	return reconstruct, U[:, :num]
	
def draw_eigenfaces(eface):
	fig = plt.figure()
	for i in range(eface.shape[1]):
		ax = fig.add_subplot(np.sqrt(eface.shape[1]), np.sqrt(eface.shape[1]), i+1)
		ax.imshow(eface[:,i].reshape(64, 64), cmap='gray')
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		plt.xlabel('eigenfaces %d' %(i+1))
	plt.tight_layout()
	fig.savefig('eigenfaces.png', dpi=100)
	
def draw_reconstruct(eface, imgMean):
	fig = plt.figure(figsize=(16, 9))
	for i in range(eface.shape[1]):
		ax = fig.add_subplot(np.sqrt(eface.shape[1]), np.sqrt(eface.shape[1]), i+1)
		ax.imshow((eface[:,i]+imgMean).reshape(64, 64), cmap='gray')
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
	plt.tight_layout()
	fig.savefig('reconst100.png', dpi=100)

if __name__ == '__main__':
	img = readimg()
	imgMean = np.mean(img, axis=1)
	
	# for 1-1
	fig = plt.figure()
	plt.imshow(imgMean.reshape(64,64), cmap='gray')
	fig.savefig('average.png', dpi=100)
	rec, eface = pca(img, imgMean, 9)
	draw_eigenfaces(eface)
	
	# for 1-2
	rec, eface = pca(img, imgMean, 5)
	draw_reconstruct(rec, imgMean)
	
	# for 1-3
	for i in range(100):
		
		img_dif = img - np.tile(imgMean, (1, img.shape[1]))
		rec, eface = pca(img, imgMean, i+1)
		
		img_err = np.sqrt( np.mean( np.square(img_dif - rec) ) ) / 255.0
		
		print('testing for dimension %3d with error %f' %(i+1, img_err))
		
		if (img_err <= 0.01):
			break
	
	
	
