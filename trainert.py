from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import json
import os



# Load VGG16 network
model = VGG16(weights='imagenet', include_top=False)
#print(model.summary())


# Image loading/preprocessing
dataset = pd.DataFrame({'name': [], 'image': [], 'labels': []})

#Data
train_data = []
img_path = './valdata/'
for file in os.listdir(img_path):
	img = image.load_img(img_path + file, target_size=(200, 200))
	file = file.replace(".jpg", "")
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	dataset = dataset.append({'name': file, 'image': x, 'labels': 'None'}, ignore_index=True)
	#train_data.append(x)


print(dataset)





'''
#Labels

#CONTINUE HERE
train_labels = []
with open('validation.json') as json_data:
    d = json.load(json_data)

images = d['annotations']

for el in images:
	print(el['imageId'])

'''

'''

#Append input layers before the pretrained network
input = Input(shape=(200,200,3),name = 'image_input')
model = model(input)

#append last layers to the network
model = Flatten(name='flatten')(model)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dense(120, activation='softmax', name='predictions')(model)

finalModel = Model(input=input, output=model)

print(finalModel.summary())

features = finalModel.predict(x)
print(features)


'''