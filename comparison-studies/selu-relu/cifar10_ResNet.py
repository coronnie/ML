'''
Adapted from keras example cifar10_cnn.py
train resnet-18 on the CIFAR10 small images dataset


'''
from __future__ import print_function
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras import backend

import numpy as np
from ML.models import ResNet
from ML.data import cifar



lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the cifar10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
dataloader = cifar.CifarLoader('../../data/cifar-10-batches-py/')
X_train, y_train = dataloader.load_train()
X_test, y_test = dataloader.load_test()
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

if backend.image_data_format() == 'channels_last':
	X_train = X_train.transpose(0, 2, 3, 1)
	X_test = X_test.transpose(0, 2, 3, 1)

# convert class vectors to binary class matrices
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normarlize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

model = ResNet.ResNet18((img_rows, img_cols, img_channels), nb_classes, dropout=0.1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not data_augmentation:
	print('Not using data augmentation')
	model.fit(X_train, Y_train, batch_size=batch_size,
			  nb_epoch=nb_epoch,validation_data=(X_test, Y_test),
			  shuffle = True, callbacks=[lr_reducer, early_stopper, csv_logger])
else:
	print('Using data augmentation')
	# with preprocessing and realtime data augmentation
	datagen = ImageDataGenerator(
		featurewise_center=False, # set input mean to 0 over the dataset
		samplewise_center=False, # set each sample mean to 0
		featurewise_std_normalization=False, # divide inputs by std of the dataset
		samplewise_std_normalization=False, # divide each input by its std
		zca_whitening=False, # apply ZCA whitening
		rotation_range=0, # ramdomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1, # randomly shift images horizontally (fraction of the total width)
		height_shift_range=0.1, # randomly shift images vertically (fraction of the total height)
		horizontal_flip=True, # randomly flip images horizontally
		vertical_flip=False) # randomly flip images vertically

	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(X_train)

	# fit the model on the batches generated by datagen.flow()
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
						steps_per_epoch=X_train.shape[0] // batch_size,
						validation_data=(X_test, Y_test),
						epochs=nb_epoch, verbose=1, max_queue_size=100,
						callbacks=[lr_reducer, early_stopper, csv_logger])

