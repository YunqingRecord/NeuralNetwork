from keras.models import load_model

model = load_model('CNN_MNISTtanh.h5')
print(model.summary())
for layer in model.layers:
        for weight in layer.weights:
            print(weight.name, weight.shape, weight.value)
print(model.get_weights())