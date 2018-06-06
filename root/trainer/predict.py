import keras
from train import read_image, LABELS
import pandas as pd
import time
import numpy as np
from tensorflow.python.lib.io import file_io

data_dir = 'gs://fashiondataset'
test_dir = 'gs://fashiondataset/test'

files = file_io.list_directory(test_dir)

file_io.copy('gs://fashiondataset/resnet_oud.h5', 'model.h5')

model = keras.models.load_model('model.h5')
gen = (np.array([np.asarray(read_image((224, test_dir, fn), train=False))]) for fn in files)
preds = model.predict_generator(gen, steps=len(files), verbose=1)
df = pd.DataFrame({'id': [f.split('.')[0] for f in files], 'predicted': [' '.join(map(str, s)) for s in preds]})

df.to_csv('predictions.csv', index=False)

timeNow = str(time.time())

file_io.copy('predictions.csv', 'gs://fashiondataset/predictions' + timeNow + '.csv')
