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

print(np.unique(data))


#labels = labels[labels == 3 or labels ==7]

class Gradient_Descent():
    def __init__(self,data):
        self.data = data
        self.weights = np.random.normal(scale=0.1,size=785) #sigma = 0.1, so the argument of the sigmoid function does not too big
        self.N = 12396
        self.probabilities = np.zeros(self.N)
        self.learning_rate = 0.1
        self.momentum_term = 0.1
        self.previous_gradients = np.zeros(785)
        self.gradients = np.zeros(785)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1*x))

    def calc_prop(self):
        mapping = np.vectorize(self.sigmoid)
        self.probabilities = mapping(np.dot(self.data[:,:-1],self.weights))

    def calc_error(self):
        error = self.data[:,-1]*np.log(self.probabilities) + (1-self.data[:,-1])*np.log(1-self.probabilities)
        return -1/self.N*np.sum(error)

    def gradient(self):
        a = 1/self.N*np.dot((self.probabilities-self.data[:,-1]),data[:,:-1])
        return a

    def update_weights(self):
        self.gradients = self.gradient()
        self.weights += -self.learning_rate*self.gradients

    def update_with_momentum(self):
        self.previous_gradients = self.gradients
        self.gradients = self.gradient() + self.momentum_term*self.previous_gradients
        self.weights += -self.learning_rate * self.gradients


    def iteration(self):
        print(self.calc_error())
        self.calc_prop()
        #self.update_weights()
        self.update_with_momentum()


if __name__ == '__main__':
    Test = Gradient_Descent(data)
    i = 0
    while True:
        Test.iteration()
        i += 1
        print(i)

