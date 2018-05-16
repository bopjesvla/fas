import os
import json

DATA_folder = '.'
image_folder = './root/data/train/'

with open(DATA_folder + '/train.json') as json_data:
    labels = {d['imageId']:d['labelId'] for d in json.load(json_data)['annotations']}

for fn in os.listdir(image_folder):
    idx = fn.split('.')[0]
    string = 'id_' + str(idx) + '_labels_['
    for l in labels[idx]:
        string += l + ', '
    string = string[:-2] + '].jpg'
    os.rename(image_folder + fn, image_folder + string)   
    
