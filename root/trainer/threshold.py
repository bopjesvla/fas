import pandas as pd
from train import LABELS
import numpy as np
import os

dirname, filename = os.path.split(os.path.abspath(__file__))
data_dir = os.path.join(dirname, '../../data')
df = pd.read_csv(data_dir + '/predictions.csv').dropna()

# v = df['id'].values
# print([i for i in range(40000) if i not in v])

predictions = np.array([[float(f) for f in p.split(' ')] for p in df['predicted'].values])
predictions = predictions > 0
all_labels = np.arange(1, LABELS+1)
predictions = [all_labels[s] for s in predictions]

image_ids = df['id'].astype(int)
label_ids = [' '.join(map(str, s)) for s in predictions]

submit = pd.DataFrame({'image_id': image_ids, 'label_id': label_ids})

submit.to_csv(data_dir + '/submit.csv', index=False)
