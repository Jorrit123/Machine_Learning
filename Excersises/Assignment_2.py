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

#LAYER CLASS: EVERY LAYER IS AN INSTANTIATION OF THIS CLASS

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
        self.errors = np.zeros((self.network.batch_size,self.size))

    def set_previous_layer(self,layer):
        self.previous_layer = layer

    def set_next_layer(self, layer):
        self.next_layer = layer
        self.weights = np.random.normal(size=(self.size,self.next_layer.size))

    def is_last_layer(self):
        return self.is_outputlayer

    def calculate_values_next_layer(self):
        temp_values = np.dot(self.z_values,self.weights)
        return temp_values

    def set_values(self,values):
        self.a_values = values
        activation_function = 'softmax' if self.is_last_layer() else 'sigmoid'
        self.z_values = self.activation_function(self.a_values,activation_function)
        self.avg_a_values += self.a_values/self.network.batch_size
        self.avg_z_values += self.z_values/self.network.batch_size

    def set_errors(self,errors):
        self.errors = np.sum(errors, axis=0) / self.network.batch_size
        # self.errors = errors

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
            # deltas = np.dot(self.errors,(self.avg_z_values*(1-self.avg_z_values)[:,None]))
            # self.deltas = np.sum(deltas,axis=0)/self.network.batch_size
            self.deltas = self.errors*(self.avg_z_values*(1-self.avg_z_values))
        else:
            self.deltas = (np.dot(self.weights, self.next_layer.deltas))*(self.avg_z_values*(1-self.avg_z_values))

        if self.previous_layer is not None:
            self.previous_layer.calculate_deltas()

    def update_weights(self):
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
    def __init__(self,sizes,train_data,train_labels,test_data,test_labels,batch_size,lr):
        self.layers = []
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.batch_indices = np.arange(batch_size)
        self.lr = lr
        self.batch_counter = 0
        #self.old_corect = []

        #input layer
        self.layers.append(Layer(train_data[1,:].size,network=self))
        #intermediate layers
        for s in sizes:
            self.layers.append(Layer(s,network=self))
        #output layer ( 3 and 7)
        self.layers.append(Layer(10 if full else 2,network=self,is_outputlayer=True))

        #set next layer
        for i in range(len(self.layers)-1):
            self.layers[i].set_next_layer(self.layers[i+1])

        #set previous layer
        for i in range(1,len(self.layers)):
            self.layers[i].set_previous_layer(self.layers[i-1])

    def create_batch(self,test=False):
        if self.batch_counter + self.batch_size > self.train_data[:, 1].size:
            self.batch_counter = 0
        self.batch_indices = np.arange(self.batch_counter, self.batch_size + self.batch_counter)
        self.batch_counter += self.batch_size

        self.batch_data = self.train_data[self.batch_indices]
        self.batch_labels = self.train_labels[self.batch_indices]

    def feed_forward(self):
        self.layers[0].clean_up()
        self.create_batch()
        predicitions = []
        for index in self.batch_indices:
            self.layers[0].set_values(train_data[index, :])
            predicitions.append(self.layers[0].forward_pass())
        errors = predicitions - self.batch_labels
        self.layers[-1].set_errors(errors)

    def back_propagate(self):
        self.layers[-1].back_propagate()

    def calculate_error(self,test=False):
        correct_indices = []
        correct_predictions = 0
        if test:
            data = self.test_data
            labels = self.test_labels
        else:
            data = self.train_data
            labels = self.train_labels

        for i in range(data[:,0].size):
            self.layers[0].set_values(data[i, :])
            temp_prediction = self.layers[0].forward_pass()
            temp = temp_prediction.copy()
            temp_prediction[np.where(temp_prediction == np.max(temp_prediction))] = 1
            temp_prediction[np.where(temp_prediction != np.max(temp_prediction))] = 0

            if np.array_equal(temp_prediction,labels[i]):
                correct_predictions += 1*np.max(temp_prediction)
                correct_indices.append(i)
        return 1 - correct_predictions/data[:,1].size


train = []
test = []
iterations = []

if __name__ == '__main__':
    batch_size = 20
    layers = [100]
    lr = 0.1
    Network = Network(layers, grote_matrix, labels_matrix, test_data, test_labels,batch_size,lr)

    start = time.time()
    for i in range(10000):
        if i%100 == 0:
            print(i)
            print("train_error")
            print(Network.calculate_error())
            print("test_error")
            print(Network.calculate_error(True))
            print("#######################################")

            train.append(Network.calculate_error())
            test.append(Network.calculate_error(True))
            iterations.append(i)
        Network.feed_forward()
        Network.back_propagate()
    end = time.time()

fig = plt.figure()
ax1 = fig.add_subplot(111)


ax1.plot(iterations, train, color='black', label='Training Error')
ax1.plot(iterations, test, color='green', label='Test Error')

name = "Batch: " + str(batch_size) + "Layers: " + str(layers) + " - Time Elapsed: " + str(round(end - start, 2))
plt.title(name)

plt.legend()
#plt.savefig(name+".png")
plt.show()