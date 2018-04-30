from __future__ import print_function


import tensorflow as tf
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

rng = np.random

train = pd.read_csv("export-TxGrowth.csv", header=None, usecols = [1,2])
train_iloc = train.iloc[1:]
matrain_x = []
value = []
print(train_iloc)
for index, rows in train_iloc.iterrows():
    train_data = train.iloc[index][1]
    matrain_x += [float(train_data)//10000]
    value += [train.iloc[index][2]]

print(train_data, value)


# Parameters
learning_rate = 0.001
training_epochs = 1000
display_step = 50

# Training Data
train_X = array(matrain_x)
train_Y = array(value)
n_samples = train_X.shape[0]
train_X = train_X.reshape(-1,1)
train_Y = train_Y.reshape(-1,1)
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
train_Y = sc.fit_transform(train_Y)

print(train_X, train_Y)


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()