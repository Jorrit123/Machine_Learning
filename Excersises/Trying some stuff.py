import numpy as np
from mnist import MNIST
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import cProfile



mndata = MNIST('samples')
images, labels = mndata.load_training()
images = np.array(images)/255
bias = np.ones(images[:,0].shape)
images = np.concatenate([images,bias[:,None]], axis=1)
labels = np.array(labels)
data = np.concatenate([images, labels[:,None]], axis=1)
data = data[(data[:,-1] == 3) | (data[:,-1] == 7)]
data[:,-1][data[:,-1]==3]=0
data[:,-1][data[:,-1]==7]=1

labels = data[:,-1]
data = data[:,:-1]

grote_matrix = np.random.normal(size=(12396,785))

for i in range(12396):
    for j in range(785):
        grote_matrix[i,j] = data[i,j]



test_images, test_labels = mndata.load_testing()
test_images = np.array(test_images)/255
test_bias = np.ones(test_images[:,0].shape)
test_images = np.concatenate([test_images,test_bias[:,None]], axis=1)
test_labels = np.array(test_labels)
test_data = np.concatenate([test_images, test_labels[:,None]], axis=1)
test_data = test_data[(test_data[:,-1] == 3) | (test_data[:,-1] == 7)]
test_data[:,-1][test_data[:,-1]==3]=0
test_data[:,-1][test_data[:,-1]==7]=1

test_labels = test_data[:,-1]
test_data = test_data[:,:-1]

weights = np.random.normal(scale=1,size=785)

start = time.time()
for i in range(100):
    a = np.dot(grote_matrix,weights)
end = time.time()
print(end-start)
def calc_prop(data, weights):
    probabilities = 1 / (1 + np.exp(-np.dot(data, weights)))
    return probabilities

def gradient(probabilities,labels,data):
    gradients = 1/785 * np.dot((probabilities - labels), data)
    return gradients

start = time.time()
for i in range(100):
    probabilites = calc_prop(grote_matrix,weights)
    weights -= gradient(probabilites,labels,data)
    a = grote_matrix.size

end = time.time()
print(end-start)



