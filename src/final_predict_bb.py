# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 13:58:22 2017

@author: SinhaU
"""

from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000 

WEIGHTS_FILE = 'final_weights_bb.h5'
ROOT_WEIGHTS_DIR = '/home/icarus/kaggle/Kaggle-Fish/model_weights'
DATA_DIR = '/home/icarus/kaggle/Kaggle-Fish/data/Final_Derived_Images_bb/'
PRED_FILE = 'final_pred_bb1.csv'
PRED_TEXT = 'final_pred_bb1_txt.txt'
#ROOT_DATA_DIR = ''

root_path = '/home/icarus/kaggle/Kaggle-Fish/output'
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

weights_path =  os.path.join(ROOT_WEIGHTS_DIR,WEIGHTS_FILE)

test_data_dir = DATA_DIR #os.path.join(ROOT_DATA_DIR, DATA_FILE)

# test data generator for prediction
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_Inception)
print (test_data_dir)
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

test_image_list = test_generator.filenames
print(len(test_image_list))
print (test_image_list[0:3])
print('Loading model and weights from training process ...')
#InceptionV3_model = load_model(weights_path)
#
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
#output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = InceptionV3_notop.output  # Shape: (8, 8, 2048)
#output = GlobalAveragePooling2D()(output)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(7, activation='softmax', name='predictions')(output)


model = Model(InceptionV3_notop.input, output)
model.load_weights(weights_path)
optimizer = SGD(lr = 0.0001, momentum = 0.9, decay = 0.0, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
# for each image
#
#==============================================================================
#==============================================================================
#  prediction = InceptionV3_model.predict_classes()
#  print prediction
#==============================================================================


predictions = model.predict_generator(test_generator, nbr_test_samples)
print type(predictions)
print predictions.shape
# insert nof colum
nof_zeros = np.zeros((nbr_test_samples))

predictions = np.insert(predictions[:nbr_test_samples],4, nof_zeros,1)
for i, in_image_name in enumerate(test_image_list):
    im_name = in_image_name.split('_p')[0] + '.jpg'
    pr = float(in_image_name.split('_p')[1][0:4])
    if i < 5:
	print ( im_name + ' pr=' + str(pr))
    if pr > 0.20 :
	nof_pred =  1.0 - pr
	predictions[i] = predictions[i] *pr 
    else :
	nof_pred = 0.8 
	predictions[i] = predictions[i]* 0.2
    predictions[i,4] = nof_pred


np.savetxt(os.path.join(root_path, PRED_TEXT), predictions)


print('Begin to write predicted file ..')
f_submit = open(os.path.join(root_path, PRED_FILE), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, in_image_name in enumerate(test_image_list):
    image_name = in_image_name.split('_p')[0] + '.jpg'
    pred = ['%.6f' % p for p in predictions[i, 0:]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('prediction file successfully generated!')
#==============================================================================
