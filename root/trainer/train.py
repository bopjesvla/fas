################################################################################
#  This script provides a general outline of how you can train a model on GCP  #
#  Authors: Mick van Hulst, Dennis Verheijden                                  #
################################################################################

from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence, image
from keras.optimizers import RMSprop
from keras.applications.vgg19 import VGG19
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
import sys

MODEL_NAME = 'resnet'
CONTINUE_TRAINING = False


# for _,_,im_names in os.walk('./res/train'):
#     for im_name in im_names:
#         with Image.open('./res/train/'+im_name) as im:
#             i += 1
#             print(i, end='\r')
#             im = modify(im, 0.05)
#             im = autocrop(im)
#             im = fixsize(im, 512)

import json
from tensorflow.python.lib.io import file_io

LABELS = 228

from keras.callbacks import Callback
from keras import callbacks as cb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


dirname, filename = os.path.split(os.path.abspath(__file__))
local_dir = os.path.join(dirname, '..', '..', 'data')
local = os.path.isdir(local_dir)

class SaveToBucket(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if not local:
            with file_io.FileIO(MODEL_NAME+'.h5', mode='rb') as input_f:
                with file_io.FileIO(data_dir + '/'+MODEL_NAME+'.h5', mode='wb+') as output_f:
                    output_f.write(input_f.read())
save_to_bucket = SaveToBucket()

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_x, val_y = zip(*self.validation_generator)
        val_predict = (np.asarray(val_x)).round()
        val_targ = np.array(val_y)
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
metrics = Metrics()

def load_data(path):
    """
    Loading the data and turning it into a pandas dataframe
    :param path: Path to datafile; Can be predefined as shown above.
    :return: pandas dataframe
    """
    with file_io.FileIO(path, 'r') as f:
        json_data = json.load(f)
    data = pd.DataFrame.from_dict(json_data)
    return data


def create_model():
    """
    In here you can define your model
    NOTE: Since we are only saving the model weights, you cannot load model weights that do
    not have the exact same architecture.
    :return:
    """
    # model = Sequential()
    # model.add(Dense(42, activation='relu'))
    # model.add((Dense(6, activation='sigmoid')))
    
    if CONTINUE_TRAINING:
        file_io.copy(data_dir + '/' + MODEL_NAME + '.h5', 'cnt.h5')
        model = load_model('cnt.h5')
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
        return model
    
    """
    if MODEL_NAME == 'vgg': 
        x = VGG16(weights='imagenet', include_top=False)
    elif MODEL_NAME == 'resnet':
        x = ResNet50(weights='imagenet', include_top=False)
        
    for layer in x.layers[1:]:
        layer.trainable = False

    input = Input(shape=(224,224,3), name='image_input')
    x = x(input)
    x = Flatten(name='flatten')(x)
    """

    x = ResNet50(weights='imagenet', include_top=False)


    #while( x.layers[-1].get_config()['name'] != "activation_40" ):
    #    x.layers.pop()

    for layer in x.layers[1:]:
        layer.trainable = False
        if layer.get_config()['name'] == 'activation_40':
            break
        
        

    input = Input(shape=(224,224,3), name='image_input')
    x = x(input)
    #x = AveragePooling2D(pool_size=(1,1),name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(LABELS, activation='sigmoid',name='predictions')(x)
    
    
    #x = Dense(4096, activation='relu', name='fc1')(x)
    #x = Dense(4096, activation='relu', name='fc2')(x)
    #x = Dense(LABELS, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

    model.summary()

    return model

if local:
    data_dir = local_dir
else:
    data_dir = 'gs://fashiondataset'

print(data_dir)

with file_io.FileIO(data_dir + '/train.json', mode='r') as json_data:
    label_dict = {d['imageId']:d['labelId'] for d in json.load(json_data)['annotations']}

def label(fn):
    id = fn.split('/')[-1].split('.')[0]
    # LOOK! WE SUBTRACT ONE! ALL LABELS ARE OFF BY ONE! e.g., 227 is actually 228.
    labels = np.array([int(l) for l in label_dict[id]]) - 1
    one_hot_labels = np.zeros((LABELS), dtype=bool)
    one_hot_labels[labels] = 1
    return one_hot_labels

def read_image(arg, train=True):
    desired_size, subdir, im_name = arg
    try:
        with file_io.FileIO(subdir + '/' + im_name, mode='rb') as im_data:
            with Image.open(im_data) as im:
                old_size = im.size
                ratio = float(desired_size)/max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])
                im = im.resize(new_size, Image.ANTIALIAS)
                new_im = Image.new("RGB", (desired_size, desired_size))
                new_im.paste(im, ((desired_size-new_size[0])//2,
                (desired_size-new_size[1])//2))

                if train:
                    return new_im, label(im_name)
                else:
                    return new_im
    except Exception as e:
        print(e)
        print("BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD IMAGE: ", im_name)
        if train:
            return -1, -1
        else:
            return np.zeros((desired_size, desired_size, 3))

# proc = subprocess.Popen ("watch -n0.1 nvidia-smi".split(), shell=False)
# proc.communicate()

def main(train_folder, test_file, job_dir):    
    model = create_model()

    def generator(subdir, batch_size):
        desired_size = 224
        file_names = [(desired_size, subdir, fn) for fn in file_io.list_directory(subdir)]
        np.random.shuffle(file_names)
        i = 0
        batch = np.zeros((batch_size, desired_size, desired_size, 3))
        labels = np.zeros((batch_size, LABELS))
        p = Pool()

        while True:
            for im, label in p.imap_unordered(read_image, file_names):
                if im == -1:
                    continue
                batch[i], labels[i] = im, label
                if i == batch_size-1:
                    yield batch, labels
                    i = 0
                else:
                    i += 1

    gen = generator(data_dir + '/train', 32)

    timeNow = time.strftime("%e%m-%H%M%S")

    # --- CALLBACK TENSORBAORD ----
    #print layers:
    #sess = K.get_session()
    #[print(n.name) for n in sess.graph.as_graph_def().node]

    embeddings_list = ['resnet50/activation_26/Relu', 'resnet50/activation_47/Relu', 'resnet50/activation_49/Relu', 'fc2/Relu', 'fc1/Relu']


    # CONTINUE WITH THIS:
    # Cant set histogram=1 because it expects validation data instead of a generator..

    save_to_bucket = SaveToBucket()

    model.fit_generator(gen, epochs=1000, steps_per_epoch=600, validation_data=generator(data_dir + '/val', 32), validation_steps=1, callbacks=[cb.ModelCheckpoint(MODEL_NAME+'.h5', save_best_only=True), cb.TensorBoard(log_dir=data_dir+'/logs/'+timeNow, batch_size=32, histogram_freq=1, embeddings_freq=0, embeddings_layer_names=embeddings_list, write_images=True) ,save_to_bucket])


    # TODO: Kaggle competitions accept different submission formats, so saving the predictions is up to you

    # Save model weights

    # Save model on google storage
    if not local:
        with file_io.FileIO(MODEL_NAME+'.h5', mode='rb') as input_f:
            with file_io.FileIO(data_dir + '/'+MODEL_NAME+'.h5', mode='wb+') as output_f:
                output_f.write(input_f.read())

    print('hallo ik ben klaar')
    exit()

if __name__ == '__main__':
    """
    The argparser can also be extended to take --n-epochs or --batch-size arguments
    """
    parser = argparse.ArgumentParser()
    
    # Input Arguments
    parser.add_argument(
      '--train-folder',
      help='GCS or local paths to training data',
      # required=True
    )

    parser.add_argument(
      '--test-folder',
      help='GCS or local paths to test data',
      # required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        # required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    print('args: {}'.format(arguments))

    main(args.train_folder, args.test_folder, args.job_dir)
