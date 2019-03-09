"""
Create by ZHAO Yunqing at 05/03/2019

Version: HKU

"""
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

start = time.time()
nb_classes = 10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()  # load data

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'validation samples')


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

stdev1 = keras.initializers.RandomNormal(mean=0.0, stddev=0.0001, seed=None)
stdev2 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
stdev3 = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)    # the elastic kernel initializer


# Functional Keras model
x = Input(shape=(32, 32, 3))
y = x
y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = BatchNormalization()(y)
y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)
y = Dropout(0.3)(y)

y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = BatchNormalization()(y)
y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)
y = Dropout(0.3)(y)

y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = BatchNormalization()(y)
y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)
y = Dropout(0.3)(y)

y = Flatten()(y)
y = Dense(units=128, activation='relu', kernel_initializer='he_normal')(y)
y = Dropout(0.5)(y)
y = Dense(units=nb_classes, activation='softmax', kernel_initializer='he_normal')(y)


model = Model(inputs=x, outputs=y, name='model')
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

nb_epoch = 100
batch_size = 64


aug_generator = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,  # set zero-mean
    featurewise_std_normalization=False,  #
    samplewise_std_normalization=False,  # divide each input by its stdev
    zca_whitening=False,
    rotation_range=0,  # randomly rotate(degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,  # randomly flip images horizontaly
    vertical_flip=False)  # randomly flip images


def plot(h, nb_epoch):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='train_acc')
    plt.plot(range(nb_epoch), val_acc, label='test_acc')
    plt.title('model accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='train_loss')
    plt.plot(range(nb_epoch), val_loss, label='test_loss')
    plt.title('model loss')
    plt.legend()
    plt.grid(True)
    plt.show()


earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=2)  # use earlystopping callback function to mornitor the traning process
aug_generator.fit(X_train)
gen = aug_generator.flow(X_train, y_train, batch_size=batch_size)

try:
    history = model.fit_generator(generator=gen, steps_per_epoch=50000//batch_size, epochs=nb_epoch, validation_data=(X_test, y_test), callbacks=[earlystopping])
except:
    model.save('CIFAR10.h5')


loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))

# plot(history, nb_epoch)

end = time.time()

print("total time used :", end - start)