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



#CONTINUE HERE
train_labels = []
with open('validation.json') as json_data:
    d = json.load(json_data)

images = d['annotations']

for el in images:
	imgId = el['imageId']
	dataLoc = dataset.loc[dataset['name'] == imgId]
	dataset.loc[dataset['name'] == '10', 'labels'] = pd.Series(['19', '34343' ,'344'])


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

finalModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

finalModel.summary()

finalModel.fit(np.array([x]), np.array([[1 for i in range(8)]]))

<<<<<<< HEAD
# features = finalModel.predict(x)
# print(features)
=======
features = finalModel.predict(x)
print(features)


'''
>>>>>>> 10186c24c41f06e54473dfb359ad765c69bde41a
