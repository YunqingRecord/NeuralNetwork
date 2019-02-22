import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model


def digit_recognition(imgfile='C:\\Users\\Yunqing\\Desktop\\SEM2\\fuzzy\\Assignment\\TestNo2.png'):
    # my_model = model_from_json(open('DNN.json').read())
    model = load_model('Mnist.h5')
    print(model.summary())

    imgData = cv2.imread(imgfile, 0)  # read the img from the filepath
    # print(imgData)
    # img0 = cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY)

    ret, img1 = cv2.threshold(imgData, 120, 255, cv2.THRESH_BINARY_INV)   # binary thresh

    print(img1)   # np array of the img1

    img = cv2.resize(img1, (28, 28), 0, 0)   # resize to the input size of the network
    arr = np.asarray(img, dtype="float32")
    print(arr.shape)
    arr = np.reshape(arr, (1, 28, 28, 1))
    print(arr.shape)

    # show the detail of the cv
    cv2.namedWindow("Image_Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Image_Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = model.predict(arr)   # output the one-hot result
    for r in results:
        r = r.tolist()  # ndarray to list
        index = r.index(max(r))
        print("Prediction Class:", index)  # out put the largest value


digit_recognition()