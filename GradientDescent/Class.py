import sys

import numpy as np
from mnist import MNIST
from scipy.optimize import minimize
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
    temp_gradients =[]

    def __init__(self,data,momentum=False,decay=False,newton=False,line_search=False,conjugate=False):
        self.pixels = 785
        self.data = data
        self.weights = np.random.normal(scale=1,size=self.pixels) #sigma = 0.1, so the argument of the sigmoid function does not too big
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
        self.line_search = line_search
        self.conjugate = conjugate
        if conjugate:
            self.line_search = True


    def limit_propbs(self, prob):
        minValue = 1e-10
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
        return limitMapping(self.probabilities)

    def calc_error(self,probs):
        error = self.data[:,-1]*np.log(probs) + (1-self.data[:,-1])*np.log(1-probs)
        if self.decay:
            error += (self.weight_decay_rate/(2*self.N))*np.sum(self.weights)**2
        return (-1/self.N)*np.sum(error)

    def gradient(self):
        gradients = 1/self.N*np.dot((self.probabilities-self.data[:,-1]),data[:,:-1])
        if self.decay or self.newton:
            gradients += (self.weight_decay_rate/self.N)*self.weights
        if self.line_search:
            self.temp_gradients = gradients
            minimum = minimize(self.line_search_error, 1).x
            gradients = gradients * minimum
            if self.conjugate:
                beta = np.dot(gradients - self.previous_gradients,gradients)/(np.linalg.norm(self.previous_gradients))**2
                self.temp_gradients = gradients - beta*self.previous_gradients

        return gradients

    def calculate_hessian(self):
        hessian = np.dot(np.multiply(self.probabilities*(1-self.probabilities),self.data[:,:-1].T),self.data[:,:-1])
        return np.linalg.inv(1/self.N*hessian + np.identity(self.pixels)*(self.weight_decay_rate/self.N))

    def update_weights(self):
        if self.momentum:
            self.previous_gradients = self.deltas
            self.deltas = self.learning_rate * self.gradient() + self.momentum_term * self.previous_gradients
        elif self.conjugate:
            self.previous_gradients = self.deltas
            self.deltas = self.gradient()
        elif self.newton:
            self.deltas = np.dot(self.calculate_hessian(),self.gradient())
        elif self.line_search:
            self.deltas = self.gradient()
        else:
            self.deltas = self.learning_rate * self.gradient()
        self.weights -= self.deltas

    def line_search_error(self,gamma):
        x = self.calc_prop(self.weights - gamma*self.temp_gradients)
        return self.calc_error(x)


    def run_iteration(self):
        self.probabilities = self.calc_prop()
        self.update_weights()



if __name__ == '__main__':
    Test = Gradient_Descent(data,False,False,False,False,True)
    i = 0
    # Test.run_iteration()
    # print(Test.line_search_error(0.1))

    while True:
        for x in range(0,1):
            Test.run_iteration()
            i += 1
        print(i)
        print(Test.calc_error(Test.probabilities))

