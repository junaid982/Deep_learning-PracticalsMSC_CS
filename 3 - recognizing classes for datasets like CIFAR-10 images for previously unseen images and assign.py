# AIM Implement deep learning for recognizing classes for datasets like CIFAR-10 images for
#previously unseen images and assign them to one of the 10 classes.

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D
from keras.datasets import cifar10


import matplotlib.pyplot as plt


(train_images,train_labels),(test_images,test_labels)=cifar10.load_data()

train_images.shape

test_images.shape

train_images,test_images=train_images/255.0,test_images/255.0


class_names=['airoplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i])
  plt.xlabel(class_names[train_labels[i][0]])

cnmodel=Sequential()

cnmodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
cnmodel.add(MaxPooling2D((2,2)))
cnmodel.add(Conv2D(64,(3,3),activation='relu'))
cnmodel.add(MaxPooling2D(2,2))
cnmodel.add(Conv2D(64,(3,3),activation='relu'))


from keras.models.sharpness_aware_minimization import Model
cnmodel.add(Flatten())
cnmodel.add(Dense(64,activation='relu'))
cnmodel.add(Dense(10,activation='softmax'))

cnmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

train_labels[1]

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)

train_labels[6]

cnmodel.fit(train_images,train_labels,epochs=3,batch_size=256)

test_labels=to_categorical(test_labels)

train_labels[6]

predictions=cnmodel.predict(test_images)
for i in range(3):
  print(predictions[i])
  print(test_labels[i])

  
