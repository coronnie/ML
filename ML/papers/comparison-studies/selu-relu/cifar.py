import os
import sys
import gzip
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
			# for i in xrange(1, 6):
			# 	file = os.path.join(self.path, 'data_batch_'+str(i))
			# 	with open(file, 'rb') as fo:
			# 		fdict = cPickle.load(fo)
			file_train = os.path.join(self.path ,'data_batch_1')
			with open(file_train, 'rb') as fo:
				fdict = cPickle.load(fo)
		data = fdict['data']
		data = data.reshape(data.shape[0], 3, 32, 32)
		labels = fdict['labels']
		return data, labels

	def load_test(self):
		if not os.path.exists(self.path):
			print "file not found"
			exit(1)
		else:
			# for i in xrange(1, 6):
			# 	file = os.path.join(self.path, 'data_batch_'+str(i))
			# 	with open(file, 'rb') as fo:
			# 		fdict = cPickle.load(fo)
			file_test = os.path.join(self.path ,'test_batch')
			with open(file_test, 'rb') as fo:
				fdict = cPickle.load(fo)

		data = fdict['data']
		data = data.reshape(data.shape[0], 3, 32, 32)
		labels = fdict['labels']
		return data, labels


dataloader = CifarLoader('../../../cifar-10-batches-py/')
cifar_data_test, labels_test = dataloader.load_test()
print cifar_data_test.shape
