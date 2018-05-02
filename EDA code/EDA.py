import pandas as pd
import numpy as np
import json

data_path= 'D:\Universiteitsmap jaar 4\Machine Learning practice 2'
train={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
labels = np.zeros(228)
annotations=train['annotations']

 #Print an example label out to showcase how to get to a single label.
print(train['annotations'][1])
print(train['annotations'][1]['labelId'][1])

#Gets how often each label is present in the dataset
for anno in annotations:
    for label in anno['labelId']:
        labels[int(label)-1]+=1
print np.amax(labels)
print np.argmax(labels)+1

#Shows per label how often it occured with other labels.
examinedLabel= 1 #examined label allows for easy comparison with how often a
#label occurs, and how often others occur with it.
#The examined label is examinedLabel+1
labelsCross = np.zeros((228,228))
for anno in annotations:
    for label in anno['labelId']:
        for otherlabel in anno['labelId']:
            labelsCross[int(label)-1,int(otherlabel)-1]+=1
print labels[examinedLabel]
print labelsCross[examinedLabel,:]
# Turns the one that always happened with the label (aka, the label itself) to zero.
labelsCross[examinedLabel,np.argmax(labelsCross[examinedLabel,:])]=0
print np.amax(labelsCross[examinedLabel,:])
print np.argmax(labelsCross[examinedLabel,:])
#Alt 3 to create ##, alt 4 to remove.

#Code for checking which image has the most labels.
bestID=-1
bestTotal=0
for anno in annotations:
    length=len(anno['labelId'])
    if length>bestTotal:
        bestID=int(anno['imageId'])
        bestTotal=length
print bestID
print bestTotal
