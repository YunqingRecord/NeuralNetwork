import keras
from keras import Sequential
from keras.layers import AveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adam, adadelta
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
import time
start = time.time()


M_class = 10         # number of class ranging (0~9)
M_epoch = 200
M_batch_size = 64   # 2^7


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train -= 128   # scaling the data, divide them by 255, since value of pixel ranges in (0, 255)
x_train /= 255.
x_test -= 128
x_test /= 255.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)  # (data_size, width, height,channel)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)  # (data_size, width, height,channel)

y_train = to_categorical(y_train, M_class)  # One Hot the label of the Mnist Images
y_test = to_categorical(y_test, M_class)

stdev1 = keras.initializers.RandomNormal(mean=0.0, stddev=0.0001, seed=None)
stdev2 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
stdev3 = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)


model = Sequential()

model.add(Conv2D(input_shape=(32, 32, 3), filters=64,  kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=stdev1))
model.add(Conv2D(filters=64,  kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=stdev2))
model.add(MaxPooling2D(pool_size=2, strides=2, padding="valid"))

model.add(Conv2D(filters=128,  kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=stdev2))
model.add(Conv2D(filters=128,  kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=stdev2))
model.add(MaxPooling2D(pool_size=2, strides=2, padding="valid"))

model.add(Conv2D(filters=256,  kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=stdev3))
model.add(Conv2D(filters=256,  kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=stdev3))
model.add(MaxPooling2D(pool_size=2, strides=2, padding="valid"))

model.add(Flatten())

model.add(Dense(128, kernel_initializer=stdev3))
# model.add(LeakyReLU(alpha=0.3))
# model.add(BatchNormalization())

model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10, kernel_initializer=stdev3))
# model.add(Activation('softmax'))


# using sgd optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# sgd = SGD(lr=0.5)

model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
model.summary()


# call back the performance using history.history
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=M_batch_size, epochs=M_epoch, shuffle=True)
model.evaluate(x_test, y_test, batch_size=128)
loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss)
print('accuracy:', accuracy)


# Plot the loss of the model
plt.figure(1)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"])

# Plot the acuuracy of the model
plt.figure(2)
plt.plot(history.history['acc'], 'g')
plt.plot(history.history['val_acc'], 'y')
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train_acc", "val_acc"], loc="upper right")

plt.show()


