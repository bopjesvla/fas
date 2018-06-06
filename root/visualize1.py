from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from multiprocessing import Pool
import sklearn
import argparse
import subprocess
import os
from PIL import Image
import time


import time
from keras.preprocessing import image, sequence
from keras import backend as K


LABELS = 228

layer_name = 'conv1'



def create_model(base_net):
    """
    In here you can define your model
    NOTE: Since we are only saving the model weights, you cannot load model weights that do
    not have the exact same architecture.
    :return:
    """
    # model = Sequential()
    # model.add(Dense(42, activation='relu'))
    # model.add((Dense(6, activation='sigmoid')))
    
    
    if base_net == 'vgg': 
        x = VGG16(weights='imagenet', include_top=False)
    elif base_net == 'resnet':
        x = ResNet50(weights='imagenet', include_top=False)
        
    for layer in x.layers[1:]:
        layer.trainable = False

    input = Input(shape=(256,256,3), name='image_input')
    x = x(input)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(LABELS, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x



def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())



model = create_model('resnet')
model.load_weights('resnet.h5', by_name=True)

#--------------------------- visualization begins

img_width = 256
img_height = 256

# this is the placeholder for the input images
input_img = model.input


res = model.layers[1]
layer_dict = dict([(layer.name, layer) for layer in res.layers[1:]])


# ------------------------------- test

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations


