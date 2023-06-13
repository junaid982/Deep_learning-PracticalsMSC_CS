from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=100 , noise=0.2 , random_state=1)

X.shape

y.shape

n_train = 30

x_train , x_test = X[:n_train] , X[n_train:]

x_train.shape

x_test.shape

y_train , y_test = y[:n_train],y[n_train:]

y_train.shape

y_test.shape

m = Sequential()
m.add(Dense(500 , input_dim = 2 , activation='relu'))

m.add(Dense(1 , activation='sigmoid'))

m.compile(loss='binary_crossentropy' , optimizer='adam' , metrics = 'accuracy')

history = m.fit(x_train , y_train , validation_data = (x_test , y_test) , epochs=4000)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'] , label='train')
plt.plot(history.history['val_accuracy'] , label='validation')
plt.legend()
