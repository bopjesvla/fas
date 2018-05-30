from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random

classifiers = []
for num in range(228):
    logreg = LogisticRegression()
    classifiers.append(logreg)

data_path= 'D:\Universiteitsmap jaar 4\Machine Learning practice 2'
train={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
annotations=train['annotations']
trainingY = []
limit=100
for index in range(228):
    dataLabels=[]
    counter=0
    for anno in annotations:
        inImage=False
        for label in anno['labelId']:
            if int(label)-1==index:
                dataLabels.append('yes')
                inImage=True
                break
        if not inImage:
            dataLabels.append('no')
        if counter>=limit:
            dataLabels.append('yes')
            break
        counter+=1
    trainingY.append(dataLabels)
print(len(trainingY))
print(len(trainingY[0]))

#Random generated x data
trainingX = []
for runningOutOfNames in range(228):
    counterRan=0
    dataX=[]
    for anno in annotations:
        modelOne = random.uniform(0,1)
        modelTwo = random.uniform(0,1)
        variables=[modelOne,modelTwo]
        dataX.append(variables)
        if counterRan>=limit:
            dataX.append(variables)
            break
        counterRan+=1
    trainingX.append(dataX)
print(len(trainingX))
print(len(trainingX[0]))

predictions=[]
for predictLabel in range(228):
    classifiers[predictLabel].fit(trainingX[predictLabel],trainingY[predictLabel])
    predSingleLabel =[]
    for dataPoint in trainingX[predictLabel]:
        predSingleLabel.append(classifiers[predictLabel].predict([dataPoint]))
    predictions.append(predSingleLabel)
print(predictions)

