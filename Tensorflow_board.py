# Buliding a sample of nerual network

import _future_ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a function of layer
def add_layer(inputs,in_size, out_size, activation_function=None):
  with tf.name_scope('layer'):
    with tf.name_scope('weights'):
      Weights = tf.Varibles(tf.random_normal([in_size, out_size]), name='W')
    with tf.name_scope('biases'):
      biases = tf.Varibles(tf.zeros([1, out_size]) + 0.1)
    with tf.name_scope('Wx_plus_b'):
      Wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

## Set up data and structure
# Make up some real data
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# Define placeholder for input to network
with tf.name_scope('input'):
  xs = tf.placeholder(tf.float32, [None,1], name='x_input')
  ys = tf.placeholder(tf.float32, [None,1], name='y_input')

# Add hidden layer
L1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# Add output layer
prediction = add_layer(L1, 10, 1, activation_function=None)

# The error between prediction and real data
learning_rate = 0.1
diff = tf.square(ys-prediction)
with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(diff, 
            reduction_indices=[1]))
            
with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  
sess = tf.ession()
# Save the imformation in graph
writer = tf.train.SummaryWriter(" ".sess.graph)

init = tf.global_variables_initializer()
see.run(init)

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs
