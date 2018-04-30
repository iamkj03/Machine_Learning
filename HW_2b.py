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
column_result = []
max_column = []

for o in range(5):
    result = []


    for z in range(10):
        # shuffle
        data_shuffle = shuffle(data)
        print("data_shuffle")
        print(data_shuffle)
        # dropping column
        drop_data_shuffle = data_shuffle.tolist()
        for p in drop_data_shuffle:
            del p[o]
        drop_data_shuffle = np.array(drop_data_shuffle)
        print(drop_data_shuffle)
        #80%
        data_trainsize = int(drop_data_shuffle.shape[0]*0.8)

        #training data
        train_data = drop_data_shuffle[:data_trainsize]
        train_features = train_data[:, :4]
        train_labels = train_data[:, -1:]-1

        #test data
        test_data = drop_data_shuffle[data_trainsize:]
        test_features = test_data[:, :4]
        test_labels = test_data[:, -1:]-1

        # Parameters
        learning_rate = 0.5
        training_epochs = 5000
        batch_size = 25
        display_step = 1

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 4]) # shape 240*5=1200
        y_ = tf.placeholder(tf.int64, [None, 1]) # 1-3 digits recognition => 3 classes
        y_hot = tf.contrib.layers.one_hot_encoding(y_, 3)

        # Set model weights
        W = tf.Variable(tf.zeros([4, 3]))
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
                # Start training
                # Training cycle

                for j in range(n_batches):

                    _, c = sess.run([train_step, cross_entropy], feed_dict={x: train_features, y_: train_labels})
                    avg_c += c / n_batches
                if i % display_step == 0:
                    print(o+1, "th deleted.", z+1, "th accuracy calculating.", "Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(avg_c), "W=", sess.run(W), "b=", sess.run(b))

            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), y_)
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({x: test_features, y_: test_labels}))
            result = result + [accuracy.eval({x: test_features, y_: test_labels})]
            max_result = max(result)

    column_result = column_result + [result]
    max_column = max_column + [max_result]

for m in range(5):
    for n in range(10):
        print(m+1, "th column deleted. ", n + 1, "th result is ", column_result[m][n])

for l in range(5):
    print(l+1, "th feature removed result is ", max_column[l])
