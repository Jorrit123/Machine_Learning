import numpy as np
from mnist import MNIST
from scipy.optimize import minimize
#import matplotlib.pyplot as plt

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

test_images, test_labels = mndata.load_testing()
test_images = np.array(test_images)/255
test_bias = np.ones(test_images[:,0].shape)
test_images = np.concatenate([test_images,test_bias[:,None]], axis=1)
test_labels = np.array(test_labels)
test_data = np.concatenate([test_images, test_labels[:,None]], axis=1)
test_data = test_data[(test_data[:,-1] == 3) | (test_data[:,-1] == 7)]
test_data[:,-1][test_data[:,-1]==3]=0
test_data[:,-1][test_data[:,-1]==7]=1


class Layer():


    def __init__(self,size,is_outputlayer):
        self.previous_layer = None
        self.next_layer = None
        self.size = size
        self.is_outputlayer - is_outputlayer
        self.weights = np.random.normal(size=(self.size,self.next.size))
        self.values = np.zeros(self.size)

    def set_previous_layer(self,layer):
        self.previous_layer = layer

    def set_next_layer(self, layer):
        self.next_layer = layer

    def is_last_layer(self):
        return self.is_outputlayer

    def calculate_values_next_layer(self):
        temp_values = np.dot(self.values,self.weights)
        temp_values[temp_values < 0] = 0
        return temp_values

    def set_values(self,values):
        self.values = values
        if self.is_last_layer():
            print(self.values)


    def forward_pass(self):
        if self.next.is_last_layer:
            print
            return
        next_values = self.calculate_values_next_layer()
        self.next.set_values(next_values)
        self.next.forward_pass()


class Network():
    def __init__(self,sizes):
        self.layers = []

        #input layer
        self.layers.append(Layer(self.data[1,:].size))
        for s in sizes:
            self.layers.append(Layer(s))
        #output layer ( 3 and 7)
        self.layers.append(Layer(2))

        #set next layer
        for i in (self.layers.size-1):
            self.layers[i].set_next_layer(self.layers[i+1])

        #set previous layer
        for i in range(1,self.layers.size):
            self.layers[i].set_previous_layer(self.layers[i-1])


    def feed_forward(self,index):
        self.layers[0].set_values(self.data[index,:])
        self.layers[0].forward_pass()



if __name__ == '__main__':
    Network = Network(2,3)
    Network.feed_forward(1)