################################################################################
#  This script provides a general outline of how you can train a model on GCP  #
#  Authors: Mick van Hulst, Dennis Verheijden                                  #
################################################################################

from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence, image
from keras.applications.vgg19 import VGG19
import sklearn
import argparse

import json
from tensorflow.python.lib.io import file_io


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

    model = VGG19(weights='imagenet')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def main(train_folder, test_file, job_dir):
    model = create_model()
    # model.fit(X_train, y_train, nb_epoch=1, batch_size=32, verbose=2)
    gen = image.ImageDataGenerator()
    x = gen.flow_from_directory('../data')
    model.fit_generator(x, steps_per_epoch=10)
    score, accuracy = model.evaluate(X_validation, y_validation)
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
