#
# almost all data's type in tensorflow is float32

from __future__ import print_function
import tensorflow as tf
import numpy as np

##
# Create data 
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# Create tensorflow structure start
Weights = tf.Varible(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Varible(tf.zeros([1]))
y = Weights * x_data + biases

learning_rate = 0.1
loss = tf.reduce_mean(tf.square(y-y_data))
optimizr = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

#
inti = tf.initialize_all_variables()
sess = tf.Session
see.run(init)

for step in range(200):
    see.run(train)
    if step % 20 == 0:
      print(step, sess.run(Weights), sess.run(biases))
