import os
import json

DATA_folder = './root/DATASET'
image_folder = './root/data/train'

#with open(DATA_folder + '/train.json') as json_data:
#    labels = {d['imageId']:d['labelId'] for d in json.load(json_data)['annotations']}

if False:
    for fn in os.listdir(image_folder):
        idx = fn.split('.')[0]
        string = 'id_' + str(idx) + '_labels_['
        for l in labels[idx]:
            string += l + ', '
        string = string[:-2] + '].jpg'
        os.rename(image_folder + fn, image_folder + string)   
    
if True:
    for fn in os.listdir(image_folder):
        idx = fn.split('_')[1]
        os.rename(image_folder + '/' + fn, image_folder + '/' + idx + '.jpg')
