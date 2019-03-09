'''

Create by ZHAO Yunqing at 5/3 2019

Version: HKU

'''
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


nb_classes = 10
class_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'validation samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255    # normalization process

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

model = load_model('CIFAR10_model.h5')

rand_id = np.random.choice(range(10000), size=10)

X_pred = np.array([X_test[i] for i in rand_id])
y_true = [y_test[i] for i in rand_id]    # load the true label
y_true = np.argmax(y_true, axis=1)
y_true = [class_name[name] for name in y_true]
y_pred = model.predict(X_pred)          # get the predicted result
y_pred = np.argmax(y_pred, axis=1)
y_pred = [class_name[name] for name in y_pred]  # get name using dictionary class_name

plt.figure(figsize=(15, 7))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_pred[i].reshape(32, 32, 3), cmap='gray')
    plt.title('Label is: %s\n Prediction is: %s' % (y_true[i], y_pred[i]), size=10)
    plt.xticks([])
    plt.yticks([])
plt.show()
