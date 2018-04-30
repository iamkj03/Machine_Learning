import tensorflow as tf
import pandas as pd
import re
import numpy as np
from numpy import array
from sklearn.preprocessing import StandardScaler
import math
train = pd.read_csv("password1.train", sep='delimiter', header=None)
password = []
strength = []
digit = []
upper = []
symbol = []
num = 0
upperletter = 0

def digit_func(num):
    return len(num)

def upper_func(letter):
    return sum(1 for c in letter if c.isupper())

for index, rows in train.iterrows():
    train_data = train.iloc[index][0].split('\t')
    password = password + [len(train_data[0])]
    strength = strength + [math.log(float(train_data[1]), 10)]

    numbers = re.findall("\d+", train_data[0])
    for i in numbers:
        num += digit_func(i)
    digit = digit + [num]
    num = 0

    upper += [upper_func(train_data[0])]
    symbol += [len(train_data[0]) - num - upper_func(train_data[0])]

print(password)
print(digit)
print(upper)
print(symbol)


x1_data = array(password)
x2_data = array(digit)
x3_data = array(symbol)
x4_data = array(upper)
y_data = array(strength)

y_data = y_data.reshape(-1,1)
sc = StandardScaler()
y_data = sc.fit_transform(y_data)

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data + W3 * x3_data + W4 * x4_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(W3), sess.run(W4), sess.run(b))