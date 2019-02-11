
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=(None, 784))
y_label = tf.placeholder(tf.int32, shape=(None, 10))

learning_rate = 0.001
epoch = 5000
batch_size = 128
display_step = 0.5

# Network Hidden Layer Nodes
nn_hidden1 = 45  # First Hidden Layer Neurons
nn_hidden2 = 45  # Second Hidden Layer Neurons

input_dim = 784  # MNIST dataset Input dims = 28*28
num_class = 10  # MNIST dataset classes =10

# TensorFlow Graph Construction
X = tf.placeholder("float", [None, input_dim])
Y = tf.placeholder("float", [None, num_class])

# Mark Weights and bias
weights = {'h1': tf.Variable(tf.random_normal([input_dim, nn_hidden1])),
           'h2': tf.Variable(tf.random_normal([nn_hidden1, nn_hidden2])),
           'out': tf.Variable(tf.random_normal([nn_hidden2, num_class]))
           }
biases = {'b1': tf.Variable(tf.random_normal([nn_hidden1])),
          'b2': tf.Variable(tf.random_normal([nn_hidden2])),
          'out': tf.Variable(tf.random_normal([num_class]))
          }


# Draw the TensorFlow Graph/Model
def FC_network(x):

    hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))

    hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, weights['h2']), biases['b2']))

    output_layer = tf.matmul(hidden_layer_2, weights['out']) + biases['out']
    return output_layer


# Modify the Tensor Graph, get prediction
output = FC_network(X)
prediction = tf.nn.softmax(output)

# Define the method of loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Evaluate NN model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables (i.e. assign their default value as above)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, epoch+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X:batch_x, Y:batch_y})
        if step % display_step == 0 or step == 1: # Calculate loss and acc.
            loss1, acc = sess.run([loss, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Testing Accuracy:",
                  sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                Y: mnist.test.labels}))
















