import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

data_path= 'D:\Universiteitsmap jaar 4\Machine Learning practice 2'
train={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
labels = np.zeros(228)
annotations=train['annotations']

 #Print an example label out to showcase how to get to a single label.
##print(train['annotations'][1])
##print(train['annotations'][1]['labelId'][1])
##
#Gets how often each label is present in the dataset
##for anno in annotations:
##    for label in anno['labelId']:
##        labels[int(label)-1]+=1
##print np.amax(labels)
##print np.argmax(labels)+1
##width = 1/1.5
##plt.bar(range(1,len(labels)+1), labels, width, color="blue")
##fig = plt.gcf()
##ax = plt.subplot(111)
##plt.ylabel("Frequencies")
##plt.xlabel("Labels")
##plt.title("Frequencies of the labels in training data")
##ax.set_yscale('log')
##plt.show()
##
###Shows per label how often it occured with other labels.
##examinedLabel= 65 #examined label allows for easy comparison with how often a
###label occurs, and how often others occur with it.
###The examined label is examinedLabel+1
##labelsCross = np.zeros((228,228))
##for anno in annotations:
##    for label in anno['labelId']:
##        for otherlabel in anno['labelId']:
##            labelsCross[int(label)-1,int(otherlabel)-1]+=1
##print labels[examinedLabel]
##print labelsCross[examinedLabel,:]
### Turns the one that always happened with the label (aka, the label itself) to zero.
##labelsCross[examinedLabel,np.argmax(labelsCross[examinedLabel,:])]=0
##print np.amax(labelsCross[examinedLabel,:])
##print np.argmax(labelsCross[examinedLabel,:])+1
###Alt 3 to create ##, alt 4 to remove.
##
###Code for checking which image has the most labels.
##bestID=-1
##bestTotal=0
##for anno in annotations:
##    length=len(anno['labelId'])
##    if length>bestTotal:
##        bestID=int(anno['imageId'])
##        bestTotal=length
##print bestID
##print bestTotal

#Code for counting the frequency of the amount of labels an image has.
##frequency =[]
##amountOfLabels =[]
##for anno in annotations:
##    length=len(anno['labelId'])
##    if length in amountOfLabels:
##        index=amountOfLabels.index(length)
##        frequency[index]+=1
##    else:
##        amountOfLabels.append(length)
##        frequency.append(1)
##width = 1/1.5
##plt.bar(amountOfLabels, frequency, width, color="blue")
##fig = plt.gcf()
##ax = plt.subplot(111)
##plt.ylabel("Frequencies")
##plt.xlabel("Number labels in image")
##plt.title("Frequencies of the number labels per image in training data")
##ax.set_yscale('log')
##plt.show()

#Code for checking the labels of the disabled labels
label_path= 'D:\Universiteitsmap jaar 4\Machine Learning practice 2\Git code\\fas\\failed\\train.txt'
failed_file = open(label_path,'r')
ids= []
for line in failed_file:
    intS=line.split(' ', 1)[0]
    ids.append(int(intS))
failed_file.close();
frequencyM =[]
missingLabels =[]
print 'start'
for anno in annotations:
    identity=int(anno['imageId'])
    if identity in ids:
        for label in anno['labelId']:
            label=int(label)
            if label in missingLabels:
                index=missingLabels.index(label)
                frequencyM[index]+=1
            else:
                missingLabels.append(label)
                frequencyM.append(1)
for anno in annotations:
    for label in anno['labelId']:
        labels[int(label)-1]+=1
for mlabel in missingLabels:
    index=missingLabels.index(int(mlabel))
    print([mlabel, frequencyM[index],labels[int(mlabel)-1]])

