import keras
from keras import Sequential
from keras.layers import AveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
import time
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # (data_size, width, height,channel)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)  # (data_size, width, height,channel)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model=load_model('Mnist.h5')
score=model.evaluate(x_train, y_train, batch_size=32)  # 数据加载和之前一样，同样需要resize
print(score)