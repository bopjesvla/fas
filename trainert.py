from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

# Load VGG16 network
model = VGG16(weights='imagenet', include_top=False)
print(model.summary())


# Image loading/preprocessing
img_path = './root/data/train/2.jpeg'
img = image.load_img(img_path, target_size=(200, 200))
x = image.img_to_array(img)

print(x.shape)

# x = np.array([x])
x = preprocess_input(x)


#Append input layers before the pretrained network
input = Input(shape=(200,200,3),name = 'image_input')
model = model(input)

#append last layers to the network
model = Flatten(name='flatten')(model)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dense(8, activation='softmax', name='predictions')(model)

finalModel = Model(input=input, output=model)

finalModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

finalModel.summary()

finalModel.fit(np.array([x]), np.array([[1 for i in range(8)]]))

# features = finalModel.predict(x)
# print(features)
