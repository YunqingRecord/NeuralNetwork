import keras
from keras import Sequential
from keras.layers import AveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

M_class = 10
M_epoch = 2555
M_batch_size = 128

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

y_train = to_categorical(y_train, M_class)
y_test = to_categorical(y_test, M_class)

model = Sequential()
model.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4)))
model.add(Flatten())
model.add(Dense(20))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mse", optimizer=sgd)

model.fit(x_train, y_train, validation_split=0.1, batch_size=M_batch_size, epochs=M_epoch)
model.evaluate(x_test, y_test, batch_size=128)
model.summary()


