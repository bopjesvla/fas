import json
import random as rand
import numpy as np
import sys
from itertools import chain
import collections


data = {}
pred = {}

T_mean  = 1.
F_mean  = 0.
T_sigma = 0.25
F_sigma = 0.25
N  = 228
max_imageId = 99999

tries = [0.]
while tries[-1] < 1:
    tries.append(tries[-1]+0.01)

if False:
    with open('./res/DATASET/train.json','r') as json_data:
        data = json.load(json_data)
        for image in data['annotations']:
            if image['imageId'] == str(max_imageId + 1):
                break
            print(image['imageId'],end='\r')
            arrgh = np.random.normal(F_mean,F_sigma,228).clip(min=0,max=1)
            for label in image['labelId']:
                arrgh[int(label)-1] = np.random.normal(T_mean,T_sigma,1).clip(min=0,max=1)[0]

            arrgh = np.round(arrgh,decimals=8)

            pred[image['imageId']] = [float(i) for i in arrgh.tolist()]
            
            #pred.append({'imageId':image['imageId'],'labelId':[float(i) for i in arrgh.tolist()]})
            
            
    with open('./res/predictions/pred.json','w+') as f:
        json.dump(pred,f)

#sys.exit(0)

labe = {}
pred = {}
T    = {}
dt   = 0.01


for i in range(1,N+1):
    labe[str(i)] = []
    pred[str(i)] = []
    T[str(i)]    = 0.

with open('./res/predictions/pred.json','r') as json_data:
    pred = json.load(json_data)

with open('./res/DATASET/train.json','r') as json_data:
    true_data = json.load(json_data)
    for image in true_data['annotations']:
        for l in image['labelId']:
            if int(image['imageId']) <= max_imageId:
                labe[l].append(image['imageId'])

def F1_score( t, labelId ):
    pred_T = []
    for key in pred:
        if pred[key][int(labelId)-1] > t:
            pred_T.append(key)
    overlap   = len(set(labe[labelId]).intersection(pred_T))
    precision = overlap / (len( pred_T ) + 0.001)
    recall    = overlap / (len(labe[labelId]) + 0.001)
    F1_ = 2*(precision*recall)/(precision+recall+0.001)
    return F1_

print(" label ID |    F1    |       t   ")
print("----------|----------|----------------")

for labelId_int in range(1,N+1):
    labelId = str(labelId_int)

    best = 0.
    search_list = tries
    while True:
        if len(search_list) == 1:
            best = search_list[0]
            break
        if len(search_list)==2:
            F1_1 = F1_score( search_list[0], labelId)
            F1_2 = F1_score( search_list[1], labelId)
            if F1_1 > F1_2:
                best = search_list[0]
            else:
                best = search_list[1]
            break
        else:
            print(search_list[int(len(search_list)/2)-1],end='\r')
            F1_1 = F1_score( search_list[int(len(search_list)/2)-1], labelId)
            F1_2 = F1_score( search_list[int(len(search_list)/2)],labelId)
            if F1_1 > F1_2:
                search_list = search_list[:int(len(search_list)/2)]
            else:
                search_list = search_list[int(len(search_list)/2):]

    #best = 0.
    #F1   = 0.
    #t    = 0.
    #while t < 1.:
    #    print(labelId + ' - t: ' + "%.2f"%t + '                    ',end='\r')
    #    F1_ = F1_score(t,labelId)
    #    if F1_ > F1:
    #        best = t
    #        F1   = F1_
    #    t += dt

    t = best
    F1 = F1_score(best,labelId)
    for i in range(0,0):
        print(labelId + " - t: %.2f "%t + "."*(i%25) + " "*(25-(i%25)),end='\r')
        t_ = t + np.random.normal(0,0.01)
        F1_ = F1_score(t_,labelId)
        if F1_ > F1:
            best = t_
            F1   = F1_

    T[labelId] = best
    print( " " * (8-len(labelId)) + labelId + "  |   " + "%.2f"%F1 + "   |   " + "%.4f"%best + "             ")
    

with open('./res/thresholds.json','w+') as f:
    json.dump(T,f)

