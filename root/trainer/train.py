################################################################################
#  This script provides a general outline of how you can train a model on GCP  #
#  Authors: Mick van Hulst, Dennis Verheijden                                  #
################################################################################

from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence, image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from multiprocessing import Pool
import sklearn
import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image

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

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
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

    x = VGG16(weights='imagenet', include_top=False)
    input = Input(shape=(256,256,3), name='image_input')
    x = x(input)

    # for layer in x.layers[1:]:
    #     layer.trainable = False

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(LABELS, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


dirname, filename = os.path.split(os.path.abspath(__file__))
local_dir = os.path.join(dirname, '../../data')
if os.path.isdir(local_dir):
    data_dir = local_dir
else:
    data_dir = 'gs://fashiondataset'

with open(data_dir + '/train.json') as json_data:
    label_dict = {d['imageId']:d['labelId'] for d in json.load(json_data)['annotations']}

def label(fn):
    id = fn.split('/')[-1].split('.')[0]
    # LOOK! WE SUBTRACT ONE! ALL LABELS ARE OFF BY ONE! e.g., 227 is actually 228.
    labels = np.array([int(l) for l in label_dict[id]]) - 1
    one_hot_labels = np.zeros((LABELS), dtype=bool)
    one_hot_labels[labels] = 1
    return one_hot_labels

def read_image(arg):
    desired_size, subdir, im_name = arg
    with Image.open(subdir+'/'+im_name) as im:
        old_size = im.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
        (desired_size-new_size[1])//2))

        return new_im, label(im_name)

def main(train_folder, test_file, job_dir):
    model = create_model()
    # model.fit(X_train, y_train, nb_epoch=1, batch_size=32, verbose=2)
    gen = image.ImageDataGenerator()

    def generator(subdir, batch_size):
        desired_size = 256
        file_names = [(desired_size, subdir, fn) for fn in os.listdir(subdir)]
        i = 0
        batch = np.zeros((batch_size, desired_size, desired_size, 3))
        labels = np.zeros((batch_size, LABELS))
        p = Pool()

        while True:
            for im, label in p.imap_unordered(read_image, file_names):
                batch[i], labels[i] = im, label
                if i == batch_size-1:
                    yield batch, labels
                    i = 0
                else:
                    i += 1

    # labels = {d["imageId"]: d["labelId"] for d in json.loads('train.json')["annotations"]}

    # def generate(batch_size):
    #     while True:
    #         for name in os.walk(data_dir + '/train'):

    # gen = ((x, labels(data)) for (x, _) in data)

    gen = generator(data_dir + '/train', 32)

    # for (x, y) in gen:
    #     for (img, l) in zip(x, y):
    #         if all(l[np.array([190, 106, 53, 137, 153, 74, 164, 138]) - 1]):
    #             plt.imshow(img)
    #             plt.show()

    model.fit_generator(gen, steps_per_epoch=1, validation_data=generator(data_dir + '/test', 32))
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    X_test = load_data(test_file)

    predictions = model.predict(X_test)
    # TODO: Kaggle competitions accept different submission formats, so saving the predictions is up to you

    # Save model weights
    model.save('model.h5')

    # Save model on google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO('./tmp/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

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
