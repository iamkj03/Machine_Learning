import pandas as pd
import numpy as np
from sklearn import svm
# Packages for visuals
import matplotlib.pyplot as plt
# Pickle package
import pickle
from sklearn.utils import shuffle
import seaborn as sns; sns.set(font_scale=1.2)

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
    data_label = data_shuffle[:, -1:]

    #training data
    train_data = data_shuffle[:data_trainsize]
    train_features = train_data[:, :5]
    train_labels = train_data[:, -1:]

    print(train_labels)
    #test data
    test_data = data_shuffle[data_trainsize:]
    test_features = test_data[:, :5]
    test_labels = test_data[:, -1:]
    test_labels = test_labels.transpose()
    print(test_data)
    print(test_labels)

    clf = svm.SVC()
    clf.fit(train_features, train_labels.ravel())
    a = clf.predict(test_features)
    sum = 0
    print(test_labels[0][1])
    print(a)
    for i in range(test_data.shape[0]):
        if test_labels[0][i]==a[i]:
            sum += 1

    accuracy = float(sum/test_data.shape[0])
    print(accuracy)
    result = result + [accuracy]

for i in range(10):
    print(i+1, "th accuracy is ", result[i])