from keras.utils import plot_model
from keras.models import load_model


model = load_model('cc.h5')
model.summary()

plot_model(model, to_file='C:\\Users\Yunqing\Desktop\\11.png', show_shapes='True')