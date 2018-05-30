import keras
from train import read_image, LABELS, create_model
import pandas as pd
import numpy as np
import os
from tensorflow.python.lib.io import file_io

dirname, filename = os.path.split(os.path.abspath(__file__))
data_dir = os.path.join(dirname, '../../data')
test_dir = data_dir + '/test'
files = file_io.list_directory(test_dir)

model = keras.models.load_model(data_dir + '/model.h5')
# model = create_model()

gen = np.array([np.asarray(read_image((256, test_dir, fn), train=False)) for fn in files])
scores = model.predict(gen, steps=len(files))
scores = scores > .5
all_labels = np.arange(1, LABELS+1)
scores = [all_labels[s] for s in scores]
df = pd.DataFrame({'id': [f.split('.')[0] for f in files], 'predicted': [' '.join(map(str, s)) for s in scores]})
with file_io.FileIO(data_dir + '/predictions.csv', mode='w+') as csv:
    df.to_csv(csv, index=False)
