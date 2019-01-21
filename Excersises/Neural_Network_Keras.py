
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

import time

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

# full = True -> all data
# full = False -> 3's and 7's
full = True

#PREPOCESSING TRAIN DATA
mndata = MNIST('samples')
images, labels = mndata.load_training()
images = np.array(images)/255
bias = np.ones(images[:,0].shape)
print(images.shape,bias.shape)
images = np.concatenate([images,bias[:,None]], axis=1)
labels = np.array(labels)

grote_matrix = np.random.normal(size=(60000,785))

for i in range(60000):
    for j in range(785):
        grote_matrix[i,j] = images[i,j]


labels_matrix= []
if full:
    for l in labels:
        temp = np.zeros(10)
        temp[l]=1
        labels_matrix.append(temp)
    train_data = grote_matrix


else:
    train_data = np.concatenate([grote_matrix, labels[:, None]], axis=1)
    train_data = train_data[(train_data[:, -1] == 3) | (train_data[:, -1] == 7)]

    train_labels = train_data[:, -1]
    train_data = train_data[:, :-1]

    for l in train_labels:
        temp = [1,0] if l == 3 else [0,1]
        labels_matrix.append(temp)


labels_matrix = np.array(labels_matrix)


#PREPROCESSING TEST
images, labels = mndata.load_testing()
images = np.array(images)/255
bias = np.ones(images[:,0].shape)
images = np.concatenate([images,bias[:,None]], axis=1)
labels = np.array(labels)

labels_matrix_test = []
if full:
    for l in labels:
        temp = np.zeros(10)
        temp[l]=1
        labels_matrix_test.append(temp)
    test_data = images

else:
    test_data = np.concatenate([images, labels[:, None]], axis=1)
    test_data = test_data[(test_data[:, -1] == 3) | (test_data[:, -1] == 7)]

    test_labels = test_data[:, -1]
    test_data = test_data[:, :-1]
    labels_matrix_test= []


    for l in test_labels:
        temp = [1,0] if l == 3 else [0,1]
        labels_matrix_test.append(temp)

test_labels = np.array(labels_matrix_test)


# convert class vectors to binary class matrices

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])