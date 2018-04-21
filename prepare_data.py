from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

def load_data(file_path):
	#file_path = '/home/ujjax/Documents/nips/pred-lstm/data/mnist_test_seq.npy'
	data = np.load(file_path)
	print('Data is of shape {}'.format(data.shape))
	return data

def get_batches(data, batch_size):
	length = len(data[0])
	for i in range(0,length,batch_size):
		batch = data[:,i:i+batch_size,:,:]			#[20,batch_size,64,64]
		batch_x = batch[0:10]
		batch_y = batch[10:20]
		yield (batch_x,batch_y)