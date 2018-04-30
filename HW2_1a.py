from __future__ import print_function


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

train1 = pd.read_csv("data1.txt", delimiter = '\t', header=None).values
train2 = pd.read_csv("data2.txt", delimiter = '\t', header=None).values
train3 = pd.read_csv("data3.txt", delimiter = '\t', header=None).values

data = np.concatenate([train1, train2, train3], axis = 0)
result = []

for i in range(10):
    #shuffle
    data_shuffle = shuffle(data)
    print(data_shuffle)

    #80%
    data_trainsize = int(data_shuffle.shape[0]*0.8)

    #training data
    train_data = data_shuffle[:data_trainsize]
    train_features = train_data[:, :5]
    train_labels = train_data[:, -1:]-1

    #test data
    test_data = data_shuffle[data_trainsize:]
    test_features = test_data[:, :5]
    test_labels = test_data[:, -1:]-1

    # Parameters
    learning_rate = 0.5
    training_epochs = 5000
    batch_size = 25
    display_step = 1

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 5]) # shape 240*5=1200
    y_ = tf.placeholder(tf.int64, [None, 1]) # 1-3 digits recognition => 3 classes
    y_hot = tf.contrib.layers.one_hot_encoding(y_, 3)

    # Set model weights
    W = tf.Variable(tf.zeros([5, 3]))
    b = tf.Variable(tf.zeros([3]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    # Minimize error using cross entropy
    # Gradient Descent
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=pred)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(training_epochs): # train the model n_epochs times
            avg_c = 0
            n_batches = int(data_trainsize / batch_size)
            for j in range(n_batches):

                _, c = sess.run([train_step, cross_entropy], feed_dict={x: train_features, y_: train_labels})
                avg_c += c / n_batches
        # Start training
        # Training cycle

        #for _ in range(training_epochs):


              # Run optimization op (backprop) and cost op (to get loss value)
         #   _, c = sess.run(train_step, feed_dict={x: train_features, y_: train_labels})
                # Compute average loss


            if i % display_step == 0:
                print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(avg_c), "W=", sess.run(W), "b=", sess.run(b))



        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), y_)
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: test_features, y_: test_labels}))
        result = result + [accuracy.eval({x: test_features, y_: test_labels})]

for j in range(10):
    print(j+1, "th result is ", result[j])
