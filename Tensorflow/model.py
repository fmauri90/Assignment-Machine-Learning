import input_data
import tensorflow as tf

import plotter

# MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Values that we will input during the computation
# le immagini di ingresso x sara' composto da un 2d tensor di numeri in virgola mobile. Qui si assegna una forma di [None, 784], dove 784 e' la dimensionalita' di una singola immagine MNIST    appiattita, e None indica che la prima dimensione, corrispondente alla dimensione del lotto, puo' essere di qualsiasi dimensione.
x = tf.placeholder("float", shape=[None, 784])

# il target output classes y consiste in un 2d tensor dove ogni riga e' un vettore one-hot 10-dimensionale che indica a quale digit class appartiene la MNIST image corrispondente
y = tf.placeholder("float", shape=[None, 10]) # crea nodi per target output classes

# reshape vectors of size 784, to squares of the size 28x28
x_image = tf.reshape(x, [-1,28,28,1])

"""
Initializing variables to zero, when the activation of a layer is made of ReLUs will yoeld a null gradient. This generates dead neurons -> no learning!
More precisely a ReLU is not differentiable in 0, but it is differentiable in any epsilon bubble defined around 0.
"""

# init for a weight variable
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial) #crea la variabile con il valore iniziale initial

# init for a bias variable
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#define functions for the convolution and max pooling

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#1st layer: convolution layer with max pooling
"""1st layer: convolution layer with max pooling"""
#[width, height, depth, output_size]
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd layer: convolution layer with max pooling
"""2nd layer: convolution layer with max pooling"""
#[width, height, depth, output_size]
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#3rd layer: ReLU
"""3rd layer: fully connected layer"""
#[input_size, output_size]
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
#floattening the output of the previous layer
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # matmul(h_pool2_flat, W_fc1) moltiplica h_pool2 e W_fc1

#3rd layer: add dropout
"""Add dropout"""
#using a placeholder for keep_prob will allow to turn off the dropout during testing
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#4th layer: fully connected layer
"""4th layer: fully connected layer"""
#[input_size, output_size]
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#define the cost (cross-entropy)
cross_entropy = -tf.reduce_sum(y*tf.log(y_hat)) # tf.log(y_hat) calcola il logaritmo di y_hat questo viene moltiplicato per y e in fine tf.reduce_sum aggiunge tutti gli elementi del tensor

#define the traning algorithm
#minimization of the cross-entropy (learning rate = 1e-4) with adaptive gradients.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#start a new session (for now we have not computed anything)
#if ipython, start a new InteractiveSession, it prevents garbage collection.
#all the objects will be inspectable in ipython.
sess = tf.InteractiveSession()
#Otherwise, start a new Session with sess = tf.Session().

#initiable the variables
sess.run(tf.initialize_all_variables())

#define accuracy before training (for monitoring)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#train the model
for n in range(20000):
	batch = mnist.train.next_batch(50) # Per ogni fase del ciclo si ottiene una batch di cinquanta punti dati a caso dal nostro training set.
	if n % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob: 1.0})
		print "step %d, training accurancy %g" % (n, train_accuracy)
	sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob: 0.5})	
#evaluate the prediction
print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
plotter.plot_mnist_weights(accuracy.eval())
