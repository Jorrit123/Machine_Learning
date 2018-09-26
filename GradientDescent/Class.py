import sys

import numpy as np
from mnist import MNIST
#import matplotlib as plt

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


#labels = labels[labels == 3 or labels ==7]

class Gradient_Descent():
    def __init__(self,data,momentum,decay,newton):
        self.pixels = 785
        self.data = data
        self.weights = np.random.normal(scale=0.1,size=self.pixels) #sigma = 0.1, so the argument of the sigmoid function does not too big
        self.N = 12396
        self.probabilities = np.zeros(self.N)
        self.learning_rate = 0.1
        self.momentum_term = 0.2
        self.weight_decay_rate = 0.1
        self.previous_gradients = np.zeros(self.pixels)
        self.deltas = np.zeros(self.pixels)
        self.momentum = momentum
        self.decay = decay
        self.newton = newton
        self.startGamma = 2


    def limit_propbs(self, prob):
        minValue = 1e-20
        maxValue = 1-minValue
        return min(max(minValue,prob),maxValue)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1*x))

    def calc_prop(self,x=None):
        if x is None:
            x = self.weights
        mapping = np.vectorize(self.sigmoid)
        self.probabilities = mapping(np.dot(self.data[:,:-1],x))
        limitMapping = np.vectorize(self.limit_propbs)
        self.probabilities = limitMapping(self.probabilities)

    def calc_error(self):
        error = self.data[:,-1]*np.log(self.probabilities) + (1-self.data[:,-1])*np.log(1-self.probabilities)
        if self.decay:
            error += (self.weight_decay_rate/(2*self.N))*np.sum(self.weights)**2
        return (-1/self.N)*np.sum(error)

    def gradient(self):
        gradients = 1/self.N*np.dot((self.probabilities-self.data[:,-1]),data[:,:-1])
        if self.decay or self.newton:
            gradients += (self.weight_decay_rate/self.N)*self.weights
        return gradients

    def calculate_hessian(self):
        hessian = np.dot(np.multiply(self.probabilities*(1-self.probabilities),self.data[:,:-1].T),self.data[:,:-1])
        return np.linalg.inv(1/self.N*hessian + np.identity(self.pixels)*(self.weight_decay_rate/self.N))

    def update_weights(self):
        if self.momentum:
            self.previous_gradients = self.deltas
            self.deltas = self.learning_rate * self.gradient() + self.momentum_term * self.previous_gradients
        elif self.newton:
            self.deltas = np.dot(self.calculate_hessian(),self.gradient())
        else:
            self.deltas = self.learning_rate * self.gradient()
        self.weights -= self.deltas

    def find_interval(self):
        original_gradients = self.gradient()
        self.calc_prop(self.weights-self.startGamma*original_gradients)
        start_error = self.calc_error()
        self.calc_prop(self.weights - (self.startGamma*0.1) * original_gradients)

            if self.calc_error() < start_error:
                self.search_left(original_gradients)
            else:
                self.search_right(original_gradients)


    def search_left(self, gradients):
        lastY = self.startGamma
        lastError = np.math.inf

        self.calc_prob(self.weights - (self.startGamma*0.5) * gradients)
        newError = self.calc_error()
        if newError < lastError
            lastY = lastY * 0.5
        else:
            return lastY

        while newError < lastError:
            self.calc_prob(self.weights - (self.startGamma * 0.5) * gradients)
            newError = self.calc_error()
            if newError < lastError
                lastY = lastY * 0.5
            else:
                return lastY


    def run_iteration(self):
        self.calc_prop()
        self.update_weights()


if __name__ == '__main__':
    Test = Gradient_Descent(data,False,False,True)
    i = 0
    while True:
        for x in range(0,1):
            Test.run_iteration()
            i += 1
        print(i)
        print(Test.calc_error())

