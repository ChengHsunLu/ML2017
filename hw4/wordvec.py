#!/usr/bin/env python3
#coding=utf-8

import word2vec
import numpy as np
from sklearn.manifold import TSNE
import nltk
from nltk import word_tokenize
import matplotlib.pyplot as plt
from adjustText import adjust_text

txtfile = './data/prob2/all.txt'
phrasefile = './data/prob2/all_phrase'
vecfile = './data/prob2/all.bin'
clusfile = './data/prob2/all_cluster.txt'

PUNC = ['"', ',', '.', ':', ';', '\'', '!', '?', '’', '“', '＂', '—', '-']

def train():
	word2vec.word2phrase(txtfile, phrasefile, verbose=True)
	word2vec.word2vec(phrasefile, vecfile, size=100, verbose=True)
	word2vec.word2clusters(txtfile, clusfile, 100, verbose=True)
	
def k_project(k):
	model = word2vec.load(vecfile)
	tsne = TSNE(n_components=2, random_state=0)
	trans = tsne.fit_transform(model.vectors[:k, :])
	wtag = nltk.pos_tag(model.vocab[:k])
	
	
	idx = []
	index = 0
	for (a, b) in wtag:
		punc_tag = 0
		if ((b=='JJ') or (b == 'NNP') or (b == 'NN') or (b == 'NNS')):
			
			for punc in PUNC:
				if (punc in a):
					punc_tag = 1
					break
			
			if(punc_tag == 0):
				idx = np.append(idx, index)
		
		index = index + 1
	
	wtag = np.asarray(wtag)
	idx = np.array(idx, dtype=np.uint)
	return trans, wtag, idx
	
def tags(model, k):
	wtag = nltk.pos_tag(model.vocab[:k])
	
	word_jj = []
	word_nnp = []
	word_nn = []
	word_nns = []
	
	for (a, b) in wtag:
		if (b == 'JJ'):
			word_jj = np.append(word_jj, a)
		elif(b == 'NNP'):
			word_nnp = np.append(word_nnp, a)
		elif(b == 'NN'):
			word_nn = np.append(word_nn, a)
		elif(b == 'NNS'):
			word_nns = np.append(word_nns, a)
	
	word_jj_r = word_jj.tolist()
	word_nnp_r = word_nnp.tolist()
	word_nn_r = word_nn.tolist()
	word_nns_r = word_nns.tolist()
	
	for w in word_jj:
		if(len(w) == 1):
			word_jj_r.remove(w)
		else:
			for punc in PUNC:
				if (punc in w):
					word_jj_r.remove(w)
					break
	
	for w in word_nnp:
		if(len(w) == 1):
			word_nnp_r.remove(w)
		else:
			for punc in PUNC:
				if (punc in w):
					word_nnp_r.remove(w)
					break
	
	for w in word_nn:
		if(len(w) == 1):
			word_nn_r.remove(w)
		else:
			for punc in PUNC:
				if (punc in w):
					word_nn_r.remove(w)
					break
	
	for w in word_nns:
		if(len(w) == 1):
			word_nns_r.remove(w)
		else:
			for punc in PUNC:
				if (punc in w):
					word_nns_r.remove(w)
					break
	
	word_jj = np.asarray(word_jj_r)
	word_nn = np.asarray(word_nn_r)
	word_nnp = np.asarray(word_nnp_r)
	word_nns = np.asarray(word_nns_r)	
	
	return word_jj, word_nn, word_nns, word_nnp
	
	

if __name__ == '__main__':
	#train()
	
	## projection using TSNE
	k = 700 # 500 ~ 1000
	proj_vec, wtag, idx = k_project(k)
	
	
	fig = plt.figure(figsize=(16,9))
	plt.scatter(proj_vec[idx, 0], proj_vec[idx, 1])
	
	texts = []
	for label, x, y in zip(wtag[idx, 0], proj_vec[idx, 0], proj_vec[idx, 1]):
		#plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
		texts.append(plt.text(x, y, label, size=8))
		
	adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
	fig.savefig('visualize.png', dpi=100)
	
