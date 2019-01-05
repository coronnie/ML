import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import (
	Input,
	Activation,
	Dense,
	Flatten,
	Add
)

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

def bn_relu(input):
	'''from BN -> relu'''
	norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
	return Activation('relu')(norm)

def conv_bn_relu(**conv_parameters):
	'''build a conv -> bn -> relu block'''
	filters = conv_parameters['filters']
	kernel_size = conv_parameters['kernel_size']
	strides = conv_parameters.setdefault('strides', (1, 1))
	kernel_initializer = conv_parameters.setdefault('kernel_initializer', 'he_normal')
	padding = conv_parameters.setdefault('padding', 'same')
	kernel_regularizer = conv_parameters.setdefault('kernel_regularizer', l2(1.e-4))

	def f(input):
		conv = Conv2D(filters=filters, kernel_size=kernel_size,
					  strides=strides, padding=padding,
					  kernel_initializer=kernel_initializer,
					  kernel_regularizer=kernel_regularizer
			)(input)

		return bn_relu(conv)

	return f

def bn_relu_conv(**conv_parameters):
	'''build a bn -> relu -> conv block'''
	filters = conv_parameters['filters']
	kernel_size = conv_parameters['kernel_size']
	strides = conv_parameters.setdefault('strides', (1, 1))
	kernel_initializer = conv_parameters.setdefault('kernel_initializer', 'he_normal')
	padding = conv_parameters.setdefault('padding', 'same')
	kernel_regularizer = conv_parameters.setdefault('kernel_regularizer', l2(1.e-4))

	def f(input):
		activation = bn_relu(input)
		return Conv2D(filters=filters, kernel_size=kernel_size,
					  strides=strides, padding=padding,
					  kernel_initializer=kernel_initializer,
					  kernel_regularizer=kernel_regularizer
			)(activation)

	return f

def shortcut(input, residual):
	'''shortcut between input and residual block and merges them with "sum"'''
	# expand channels of shortcut to match residual 
	# stride properly to match residual width and height
	# should be int if network architecture is correctly configured 
	input_shape = backend.int_shape(input)
	residual_shape = backend.int_shape(residual)
	stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
	stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
	equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

	shortcut = input

	# 1x1 conv if shape is different. Else identity
	if stride_width>1 or stride_height>1 or not equal_channels:
		shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
						  kernel_size=(1,1),
						  strides=(stride_width, stride_height),
						  padding = 'valid',
						  kernel_initializer = 'he_normal',
						  kernel_regularizer=l2(0.0001)
						  )(input)

	return Add()([shortcut, residual])


def residual_block(block_function, filters, repetitions, is_first_layer=False):
	'''a residual block with repeating bottleneck blocks'''
	# return a function

	def f(input):
		for i in range(repetitions):
			init_strides = (1, 1)
			if i == 0 and not is_first_layer:
				init_strides = (2, 2)
			input = block_function(filters=filters, init_strides=init_strides, is_first_block_of_first_layer=(is_first_layer and i==0))(input)

		return input

	return f

def basic_block(filters, init_strides=(1,1), is_first_block_of_first_layer=False):
	'''bottleneck architecture for > 34 layer resnet.
		follows improved scheme in http://arxiv.org/pdf/1603.05027.pdf

		returns:
			a final conv layer of filters * 4
	'''
	def f(input):
		if is_first_block_of_first_layer:
			conv_1_1 = Conv2D(filters=filters, kernel_size=(1,1),
							  strides=init_strides, padding='same',
							  kernel_initializer = 'he_normal',
							  kernel_regularizer=l2(1e-4)
				)(input)
		else:
			conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(1,1), strides=init_strides)(input)
		conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3,3))(conv_1_1)
		residual = bn_relu_conv(filters=filters*4, kernel_size=(1,1))(conv_3_3)

		return shortcut(input, residual)

	return f

def handle_dim_ordering():
	global ROW_AXIS
	global COL_AXIS
	global CHANNEL_AXIS

	if backend.image_dim_ordering == 'tf':
		ROW_AXIS = 1
		COL_AXIS = 2
		CHANNEL_AXIS = 3
	else:
		CHANNEL_AXIS = 1
		ROW_AXIS = 2
		COL_AXIS = 3

def get_block(identifier):
	if isinstance(identifier, six.string_types):
		res = globals().get(identifier)
		if not res:
			raise ValueError("Invalid {}".format(identifier))
		return res

	return identifier


class NetBuilder(object):
	"""docstring for NetBuilder"""
	def __init__(self, arg):
		super(NetBuilder, self).__init__()
		self.arg = arg

	@staticmethod
	def build(input_shape, num_outputs, block_fn, repetitions):
		if len(input_shape)!=3:
			raise Exception("Input shape must a tuple (nb_channels, nb_rows, nb_cols)")
		# permute dimension order if necessary
		#if backend.image_dim_ordering()=='tf':
		input_shape = (input_shape[1], input_shape[2], input_shape[0])

		input = Input(shape=input_shape)
		conv1 = conv_bn_relu(filters=64, kernel_size=(7,7), strides=(2,2))(input)
		pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1)

		block = pool1
		filters = 64
		for i, r in enumerate(repetitions):
			block = residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i==0))(block)
			filters *= 2

		# last activation
		block = bn_relu(block)

		# classifier block
		block_shape = backend.int_shape(block)
		pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1,1))(block)
		flatten1 = Flatten()(pool2)
		dense = Dense(units=num_outputs, kernel_initializer='he_normal', activation='softmax')(flatten1)

		model = Model(inputs=input, outputs=dense)
		return model

	@staticmethod
	def build_resnet_18(input_shape, num_outputs):
		return NetBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2])

	@staticmethod
	def build_resnet_34(input_shape, num_outputs):
		return NetBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2])

	@staticmethod
	def build_resnet_50(input_shape, num_outputs):
		return NetBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2])

	@staticmethod
	def build_resnet_101(input_shape, num_outputs):
		return NetBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2])

	@staticmethod
	def build_resnet_152(input_shape, num_outputs):
		return NetBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2])

		