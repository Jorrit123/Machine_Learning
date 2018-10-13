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
train_data = np.concatenate([images, labels[:, None]], axis=1)
train_data = train_data[(train_data[:, -1] == 3) | (train_data[:, -1] == 7)]

train_labels = train_data[:, -1]
train_data = train_data[:, :-1]
labels_matrix= []

# for l in labels:
#     temp = []
#     for  i in range(10):
#         temp.append(0)
#     temp[l]=1
#     labels_matrix.append(temp)
for l in labels:
    temp = [1,0] if l == 3 else [0,1]
    labels_matrix.append(temp)

labels_matrix = np.array(labels_matrix)
# test_images, test_labels = mndata.load_testing()
# test_images = np.array(test_images)/255
# test_bias = np.ones(test_images[:,0].shape)
# test_images = np.concatenate([test_images,test_bias[:,None]], axis=1)
# test_labels = np.array(test_labels)
# test_data = np.concatenate([test_images, test_labels[:,None]], axis=1)
# test_data = test_data[(test_data[:,-1] == 3) | (test_data[:,-1] == 7)]
# test_data[:,-1][test_data[:,-1]==3]=[1,0]
# test_data[:,-1][test_data[:,-1]==7]=[0,1]



class Layer():


    def __init__(self,size, network, is_outputlayer=False):
        self.previous_layer = None
        self.next_layer = None
        self.size = size
        self.is_outputlayer = is_outputlayer
        self.values = np.zeros(self.size)
        self.deltas = np.zeros(self.size)
        self.lr = network.lr
        self.labels = network.train_labels
        self.network = network
        self.avg_a_values = np.zeros(self.size)
        self.avg_z_values = np.zeros(self.size)
        self.avg_errors = np.zeros(self.size)

    def set_previous_layer(self,layer):
        self.previous_layer = layer

    def set_next_layer(self, layer):
        self.next_layer = layer
        self.weights = np.random.normal(scale=0.01,size=(self.size,self.next_layer.size))

    def is_last_layer(self):
        return self.is_outputlayer

    def calculate_values_next_layer(self):
        temp_values = np.dot(self.a_values,self.weights)
        return temp_values

    def set_values(self,values):
        self.a_values = values
        activation_function = 'softmax' if self.is_last_layer() else 'sigmoid'
        self.z_values = self.activation_function(self.a_values,activation_function)
        self.avg_a_values += self.a_values/self.network.batch_size
        self.avg_z_values += self.z_values/self.network.batch_size

    def set_avg_errors(self,errors):
        self.avg_errors = errors


    @staticmethod
    def activation_function(x,function):
        if function == 'relu':
            x[x < 0] = 0
            return x
        if function == 'sigmoid':
             return (np.exp(x))/(np.exp(x)+1)
        if function == 'softmax':
            values = np.zeros(x.size)
            for i in range(x.size):
                values[i] = (np.exp(x[i])/(np.sum(np.exp(x))))
            return values

    def forward_pass(self):
        if self.is_last_layer():
            return self.z_values
        next_values = self.calculate_values_next_layer()
        self.next_layer.set_values(next_values)
        return self.next_layer.forward_pass()

    def calculate_deltas(self):
        if self.is_last_layer():
            self.deltas = self.avg_errors*self.avg_z_values*(1-self.avg_z_values)
        else:
            self.deltas = (np.dot(self.weights, self.next_layer.deltas))*(self.avg_a_values*(1-self.avg_a_values))

        if self.previous_layer is not None:
            self.previous_layer.calculate_deltas()

    def update_weights(self):
        #if self.size == 3:
            #print(self.weights)
        if not self.is_last_layer():
            gradients = np.outer(self.avg_z_values,self.next_layer.deltas)
            self.weights -= self.lr*gradients
        if self.previous_layer is not None:
            self.previous_layer.update_weights()


    def back_propagate(self):
        #backpropagate deltas recursively
        self.calculate_deltas()
        #adjust weights
        self.update_weights()

    def clean_up(self):
        self.avg_a_values = np.zeros(self.size)
        self.avg_z_values = np.zeros(self.size)
        if not self.is_last_layer():
            self.next_layer.clean_up()
        else:
            self.avg_errors = np.zeros(self.size)


class Network():
    def __init__(self,sizes,train_data,train_labels,batch_size):
        self.layers = []
        self.train_data = train_data
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.batch_indices = np.arange(batch_size)
        self.lr = 1

        self.old_corect = []
        #input layer
        self.layers.append(Layer(train_data[1,:].size,network=self))
        for s in sizes:
            self.layers.append(Layer(s,network=self))
        #output layer ( 3 and 7)
        self.layers.append(Layer(2,network=self,is_outputlayer=True))

        #set next layer
        for i in range(len(self.layers)-1):
            self.layers[i].set_next_layer(self.layers[i+1])

        #set previous layer
        for i in range(1,len(self.layers)):
            self.layers[i].set_previous_layer(self.layers[i-1])

    def create_batch(self):
        self.batch_indices = np.random.randint(0,self.train_data[:,0].size, size = self.batch_size)
        self.batch = self.train_data[self.batch_indices,:]
        self.batch_labels = self.train_labels[self.batch_indices,:]

    def feed_forward(self):
        self.layers[0].clean_up()
        self.create_batch()
        predicitions = []
        for index in self.batch_indices:
            self.layers[0].set_values(train_data[index, :])
            predicitions.append(self.layers[0].forward_pass())
        errors = predicitions - self.batch_labels
        avg_errors = np.sum(errors,axis=0)/self.batch_size
        self.layers[-1].set_avg_errors(avg_errors)

    def back_propagate(self):
        self.layers[-1].back_propagate()


    def calculate_error(self):
        correct_indices = []
        correct_predictions = 0
        for i in range(self.train_data[:,1].size):
            self.layers[0].set_values(train_data[i, :])
            temp_prediction =self.layers[0].forward_pass()
            temp = temp_prediction.copy()
            temp_prediction[np.where(temp_prediction == np.max(temp_prediction))] = 1
            temp_prediction[np.where(temp_prediction != np.max(temp_prediction))] = 0

            if np.array_equal(temp_prediction,self.train_labels[i]):
                correct_predictions += 1*np.max(temp_prediction)
                correct_indices.append(i)

        #print(correct_predictions)

        #print ((set(self.old_corect) - set(correct_indices)))
        #self.old_corect = correct_indices
        print(correct_predictions/self.train_data[:,1].size)

if __name__ == '__main__':
    Network = Network([5,5], train_data, labels_matrix,2)
    for i in range(1000):
        if i%100 == 0:
            Network.calculate_error()
        Network.feed_forward()
        Network.back_propagate()

