import os
import sys
import cPickle
import numpy as np


class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, path):
		super(CifarLoader, self).__init__()
		self.path = path

	def load_train(self):
		if not os.path.exists(self.path):
			print "file not found"
			exit(1)
		else:
			num_samples = 50000
			x_train = np.empty((num_samples, 3, 32, 32), dtype='uint8')
			y_train = np.empty((num_samples,), dtype='uint8')
			for i in xrange(1, 6):
				file_train = os.path.join(self.path ,'data_batch_'+str(i))
				with open(file_train, 'rb') as fo:
					fdict = cPickle.load(fo)
				data = fdict['data']
				data = data.reshape(data.shape[0], 3, 32, 32)
				labels = fdict['labels']
				x_train[(i-1)*10000:i*10000, :, :, :] = data
				y_train[(i-1)*10000:i*10000] = labels

		return x_train, y_train

	def load_test(self):
		if not os.path.exists(self.path):
			print "file not found"
			exit(1)
		else:
			file_test = os.path.join(self.path ,'test_batch')
			with open(file_test, 'rb') as fo:
				fdict = cPickle.load(fo)

		x_test = fdict['data']
		x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
		y_test = fdict['labels']
		return x_test, y_test


