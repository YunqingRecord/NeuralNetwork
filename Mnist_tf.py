import time
import tensorflow as tf

# Import MNIST data from pointed address
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

start = time.time()


learning_rate = 0.002
epoch = 200
batch_size = 128   # 2**7 match the computer mechanism

# The No. of Iterations in each Epoch, to be decided by batch_size
iteration = mnist.train.num_examples // batch_size

display_step = 5  # Display the training process for every 5 Iteration

input_dim = 784  # MNIST dataset Input dims = 28*28, to be Avg_Pooling to 7*7
num_class = 10  # MNIST dataset classes =10

# TensorFlow Graph Construction
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Reshape the x_image make it easy to  be tuned
X_image=tf.reshape(X, [-1, 28, 28, 1])


# Input to Average Pooling, 28*28 ---> 7*7
Pool = tf.nn.max_pool(X_image, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

# Flatten the pooled X_image,  7*7 ---> (49,)
Flat = tf.layers.flatten(Pool)

# Parameters of the First Hidden Layer
W1 = tf.Variable(tf.truncated_normal([49, 128], stddev=0.1))  # Random feed in the Parameters
b1 = tf.Variable(tf.zeros([128]) + 0.1)

# Output of the First Hidden Layer
L1 = tf.nn.tanh(tf.matmul(Flat, W1) + b1)


# Parameters of the Second Hidden Layer
W2 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
b2 = tf.Variable(tf.zeros([64]) + 0.1)

# Activation of the Second Layer
L2 = tf.nn.tanh(tf.matmul(L1, W2) + b2)


# Third Layer
W3 = tf.Variable(tf.truncated_normal([64, 32], stddev=0.1))
b3 = tf.Variable(tf.zeros([32]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2, W3) + b3)


# Output Layer
W4 = tf.Variable(tf.truncated_normal([32, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3, W4) + b4)

# Modify the Tensor Graph, get prediction

# Define the method of loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Evaluate NN model by accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables (i.e. assign their default value as above)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # writer = tf.summary.FileWriter('C:\\Users\\Yunqing\\Desktop\\SEM2\\fuzzy\\Assignment\\Assignment2\\', sess.graph)
    for e in range(epoch):
        print("Current Epoch:", e)
        for step in range(1, iteration+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:  # Calculate loss and acc.
                loss1, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Current Step", step, "Testing Accuracy:",
                      sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                    Y: mnist.test.labels}))
                print("Current Step", step, "Training Accuracy:",
                      sess.run(accuracy, feed_dict={X: mnist.train.images,
                                                    Y: mnist.train.labels}))

end = time.time()
print ("Total Time used by TensorFlow Framework :", end-start)














