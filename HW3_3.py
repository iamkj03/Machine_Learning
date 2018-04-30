#refered to sujaybabruwad/LeNet-in-Tensorflow
import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard


#setting
data_augmentation = True
epochs = 100
batch_size = 128
iterations    = 391
num_classes   = 2
log_filepath  = './lenet'

accuracy_data = []
sum = 0
#trying 10 times for accuracy of this model
for p in range(10):
    print('')
    print(p+1, 'th time')

    #for shuffling and splitting data
    seed = 7
    numpy.random.seed(seed)

    # Read images and label images

    manFolder = 'imgdata/32data/man'
    womanFolder = 'imgdata/32data/woman'
    label = []

    rawdata = [image.img_to_array(image.load_img(os.path.join(manFolder, img), target_size=(32, 32), grayscale=False)) for img in os.listdir(manFolder)]

    for img in os.listdir(manFolder):
        label = label + [0]

    rawdata = rawdata + [image.img_to_array(image.load_img(os.path.join(womanFolder, img), target_size=(32, 32), grayscale=False)) for img in os.listdir(womanFolder)]

    for img in os.listdir(womanFolder):
        label = label + [1]

    X_train, X_test, Y_train, Y_test = train_test_split(rawdata, label, test_size=0.2, random_state=seed)

    #convert to array
    X_train = numpy.array(X_train)
    X_test = numpy.array(X_test)

    #shape of X train and test
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    print(rawdata, label)

    # Convert class vectors to binary class matrices.
    Y_train = keras.utils.to_categorical(Y_train, 2)
    Y_test = keras.utils.to_categorical(Y_test, 2)

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Test Set:       {} samples".format(len(X_test)))


    print("Updated Image Shape: {}".format(X_train[0].shape))
    #buile model
    def build_model():
        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
        model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
        model.add(Dense(2, activation = 'softmax', kernel_initializer='he_normal'))
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def scheduler(epoch):
        if epoch < 81:
            return 0.05
        if epoch < 122:
            return 0.005
        return 0.0005

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

       # build network
    model = build_model()
    print(model.summary())

       # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

       # start traing
    model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,callbacks=cbks,
                  validation_data=(X_test, Y_test), shuffle=True)

        # save model
    model.save('lenet.h5')

        # Score trained model.
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    accuracy_data = accuracy_data + [scores[1]]

#average of total accuracy of 10
for i in accuracy_data:
    sum = sum + i

average = sum / len(accuracy_data)

print (accuracy_data)
print (average)