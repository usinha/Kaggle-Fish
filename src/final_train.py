# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 11:14:07 2017

@author: SinhaU
"""

# binary classification
from __future__ import print_function
from keras.applications.inception_v3 import InceptionV3
import os
import cv2
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import datetime
import numpy as np

learning_rate = 0.0001
img_width = 299
img_height = 299
#nbr_train_samples = 3019
#nbr_validation_samples = 758
nbr_train_samples = 2187
nbr_validation_samples = 556
BEST_MODEL_FILE = "/home/icarus/kaggle/Kaggle-Fish/model_weights/final_weights.h5"

nbr_epochs = 30
batch_size = 32

train_data_dir = '/home/icarus/kaggle/Kaggle-Fish/data/DerivedImages_train_split'
val_data_dir = '/home/icarus/kaggle/Kaggle-Fish/data/DerivedImages_val_split'
FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
#==============================================================================
# https://gist.github.com/embanner/6149bba89c174af3bfd69537b72bca74
#==============================================================================
def preprocess_input_Inception(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).

    """
    from keras.applications.inception_v3 import preprocess_input
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)


model = Model(InceptionV3_notop.input, output)
# train all layers
########
#for layer in InceptionV3_notop.layers:
#   layer.trainable = False
#########
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model_file = BEST_MODEL_FILE
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_Inception,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=30.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
#
#val_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_Inception)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        #save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')
#
now = datetime.datetime.now
t = now()
model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        callbacks = [best_model])
print('Training time: %s'  %(now() - t))
