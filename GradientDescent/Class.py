from mnist import MNIST
from numpy.random import random
import numpy as np



mndata = MNIST('samples')
images, labels = mndata.load_training()
images = np.array(images)/255
bias = np.ones(images[:,0].shape)
images = np.concatenate([images,bias[:,None]], axis=1)
labels = np.array(labels)
data = np.concatenate([images, labels[:,None]], axis=1)
data = data[(data[:,-1] == 3) | (data[:,-1] == 7)]
data[data[:,-1]==3]=0
data[data[:,-1]==7]=1

print(data.shape)


#labels = labels[labels == 3 or labels ==7]

class Gradient_Descent():
    def __init__(self,data):
        self.data = data
        self.weights = np.random.normal(size=785)
        self.N = 12396
        self.probabilities = np.zeros(self.N)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1*x))

    def calc_prop(self):
        self.probabilities = np.array(list(map(self.sigmoid,(np.dot(self.data[:,:-1],self.weights)))))

    def calc_error(self):
        error = self.data[:,-1]*np.log(self.probabilities) + (1-self.data[:,-1])*np.log(1-self.probabilities)
        return -1/self.N*np.sum(error)

    def mean_squares(self):
        return np.sum((self.probabilities-data[:,-1])**2)

    def gradient(self):
        return np.dot((self.probabilities-self.data[:,-1]),data[:,:-1])

    def iteration(self):
        print(self.mean_squares())
        self.calc_prop()
        gradients = self.gradient()
        self.weights = self.weights - gradients



if __name__ == '__main__':
    Test = Gradient_Descent(data)
    i = 0
    while True:
        Test.iteration()
        i += 1
        print(i)

