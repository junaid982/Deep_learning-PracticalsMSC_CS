
from keras.datasets import mnist

from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from keras.models import Sequential

import  matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images.shape

test_images.shape
plt.imshow(train_images[1])

plt.imshow(test_images[1])


train_labels[1]

test_labels[1]

train_images,test_images=train_images/255.0,test_images/255.0

import numpy as np

train_images=train_images.reshape(60000,28,28,1)
test_images=test_images.reshape(10000,28,28,1)

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)

test_labels=to_categorical(test_labels)

test_labels[1]

train_labels[1]

cnnmodel=Sequential()

cnnmodel.add((Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1))))
cnnmodel.add(MaxPooling2D(2,2))
cnnmodel.add((Conv2D(32,(3,3),activation='relu')))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(10,activation='softmax'))

cnnmodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

cnnmodel

cnnmodel.fit(train_images,train_labels,epochs=3)

predictions=cnnmodel.predict(test_images)
for i in range(3):
  print(predictions[i])
  print(test_labels[i])

  
