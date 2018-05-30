import keras
from train import read_image, LABELS
import pandas as pd
import numpy as np
from tensorflow.python.lib.io import file_io

test_dir = 'gs://fashiondataset/test'

files = file_io.list_directory(test_dir)

file_io.copy('gs://fashiondataset/model.h5', 'model.h5')

model = keras.models.load_model('model.h5')
gen = np.array([np.asarray(read_image((256, test_dir, fn), train=False)) for fn in files])
scores = model.predict(gen, steps=len(files))
scores = scores > .5
all_labels = np.arange(1, LABELS+1)
scores = [all_labels[s] for s in scores]
df = pd.DataFrame({'id': [f.split('.')[0] for f in files], 'predicted': [' '.join(map(str, s)) for s in scores]})
with file_io.FileIO(test_dir + '/../predictions.csv', mode='w+') as csv:
    df.to_csv(csv, index=False)
