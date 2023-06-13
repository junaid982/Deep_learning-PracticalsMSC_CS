from google.colab import drive

drive.mount('/content/drive')


path = "drive/My Drive/Datasets"

import pandas as pd

df = pd.read_csv(path+'/pima-indians-diabetes.csv')
df.head()

#Access Only Feature

X = df.iloc[:,:-1]
X

# seperate target

y = df.iloc[:,-1]
y


# seperate training and testing dataset

from sklearn.model_selection import train_test_split

x_train , x_test , y_train ,y_test = train_test_split(X,y , test_size=0.2 , random_state = 1)

x_train


x_test

from keras.models import Sequential

from keras.layers import Dense

# create model

model = Sequential()

# create input layer

model.add(Dense(units=8 , activation = 'relu' , input_dim = 8))

# create hidden layer
model.add(Dense(units= 6 , activation = 'relu' ))

# create output layer
model.add(Dense(units=1 , activation='sigmoid'))

# compile Model
model.compile(loss = 'binary_crossentropy' , optimizer='adam' ,metrics = 'accuracy')

model.fit(x_train , y_train , epochs = 100)

p = model.predict(x_test)
p
# seperate model

model1 = Sequential()

model1.add(Dense(units=8 , activation='relu' , input_dim=8))

model1.add(Dense(units=6 , activation='relu'))

model1.add(Dense(units=1,activation = 'sigmoid'))

model1.compile(loss = 'binary_crossentropy' , optimizer='sgd' , metrics='accuracy')


model1.fit(x_train , y_train , epochs=100)

p1 = model1.predict(x_test)
p1


p[1:5]

p1[1:5]

class_labels = []

class_labels1 = []

p.size , p1.size

for i in range(p.size):
  if p[i]<0.5 :
    class_labels.append(0)
  else:
    class_labels.append(1)


for i in range(p1.size):
  if p[i]<0.5 :
    class_labels1.append(0)
  else:
    class_labels1.append(1)

from sklearn.metrics import accuracy_score

accuracy_score(y_test , class_labels)

from sklearn.metrics import accuracy_score

accuracy_score(y_test , class_labels1)


