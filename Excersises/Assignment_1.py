import sys

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

#data = data[:test_data[:,0].size]



#labels = labels[labels == 3 or labels ==7]

class Gradient_Descent():
    temp_gradients =[]

    def __init__(self,data, test_data, labels, test_labels,learning_rate,momentum_term, momentum=False,decay=False,newton=False,line_search=False,conjugate=False, batch_size = 0):
        self.pixels = 785
        self.test_data = test_data
        self.data = data
        self.labels = labels
        self.test_labels = test_labels
        self.weights = np.random.normal(scale=1,size=self.pixels)
        self.batch_size = self.data[:, 1].size
        self.batch_size = batch_size if batch_size != 0 else self.batch_size
        self.probabilities = np.zeros(self.batch_size)
        self.learning_rate = learning_rate
        self.momentum_term = momentum_term
        self.weight_decay_rate = 1
        self.previous_gradients = np.ones(self.pixels)
        self.deltas = np.ones(self.pixels)
        self.momentum = momentum
        self.decay = decay
        self.newton = newton
        self.startGamma = 2
        self.batch_indices = np.arange(self.data[:,1].size)
        self.line_search = line_search
        self.conjugate = conjugate

        self.mapping = np.vectorize(self.sigmoid)
        self.limitMapping = np.vectorize(self.limit_propbs)
        self.minValue = 1e-10
        self.maxValue = 1 - self.minValue
        self.division = 1 / self.batch_size
        if conjugate:
            self.line_search = True

    def limit_propbs(self, prob):

        return min(max(self.minValue,prob),self.maxValue)



    def sigmoid(self,x):
        return 1 / (1 + np.exp(-min(max(self.minValue,x),self.maxValue)))

    def calc_prop(self,x=None, is_test = False):
        prop_data = self.data if not is_test else self.test_data
        if x is None:
            x = self.weights
        #self.probabilities = self.limitMapping(self.mapping(np.dot(prop_data[self.batch_indices,:-1],x)))

        if self.newton:
            x = self.limitMapping(x)
        # there used to be prop_data[Batch_indices]
        self.probabilities = 1 / (1 + np.exp(-np.dot(prop_data, x)))

        return self.probabilities

    def calc_error(self,probs):
        error = self.labels*np.log(probs) + (1-self.labels)*np.log(1-probs)
        if self.decay:
            error += (self.weight_decay_rate / (2 * self.batch_size)) * np.sum(self.weights**2)
        return (-1 / self.batch_size) * np.sum(error)

    def calc_error_for_eval(self, is_test = False):
        error_data = self.data
        error_labels = self.labels
        if is_test:
            error_data = self.test_data
            error_labels = self.test_labels
        self.batch_indices = np.arange(error_data[:,1].size)
        probs = self.limitMapping(self.calc_prop(is_test = is_test))
        error = error_labels * np.log(probs) + (1 - error_labels) * np.log(1 - probs)
        # if self.decay:
        #     error += (self.weight_decay_rate / (2 * self.batch_size)) * np.sum(self.weights**2)
        return (-1 / error_data[:,1].size) * np.sum(error)

    def gradient(self):
        test = self.probabilities - self.labels
        gradients = self.division * np.dot((self.probabilities - self.labels), self.data)
        if self.decay or self.newton:
            gradients += (self.weight_decay_rate / self.batch_size) * self.weights
        if self.line_search:
            self.temp_gradients = gradients
            minimum = minimize(self.line_search_error, 1).x
            gradients = gradients * minimum
            if self.conjugate:
                beta = np.dot(gradients - self.previous_gradients,gradients)/(np.linalg.norm(self.previous_gradients))**2
                self.temp_gradients = gradients - beta*self.previous_gradients

        return gradients

    def calculate_hessian(self):
        hessian = np.dot(np.multiply(self.probabilities * (1 - self.probabilities), self.data.T),self.data)
        inverted = np.linalg.inv(1 / self.batch_size * hessian + np.identity(self.pixels) * (self.weight_decay_rate / self.batch_size))
        return inverted

        # hessian = np.dot(np.dot(self.probabilities*(1-self.probabilities),self.data),self.data.T)
        # return np.linalg.inv(self.division * hessian + np.identity(self.pixels) * (self.weight_decay_rate / self.batch_size))

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
        x = self.limitMapping(x)
        return self.calc_error(x)

    def create_batch_indices(self):
        self.batch_indices = np.random.randint(0,self.data[:,1].size, size = self.batch_size)


    def run_iteration(self):
        self.prepare_iteration()
        if self.batch_size !=self.data[:,1].size:
            self.create_batch_indices()
        self.probabilities = self.calc_prop()
        self.update_weights()



    def calc_classification_error(self, is_test = False):
        class_data = self.data if not is_test else self.test_data
        class_label = self.labels if not is_test else self.test_labels
        self.batch_indices = np.arange(class_label.size)
        self.probabilities = self.calc_prop(is_test=is_test)
        guess = np.round(self.probabilities)
        guess = list(map(bool, guess))
        labels = list(map(bool,class_label))
        right_guess = np.array(guess) == np.array(labels)
        return 1-np.array(right_guess).sum()/class_data[:,1].size

    def prepare_iteration(self):
        self.batch_indices = np.arange(self.data[:, 1].size)

def main(lr,mt,bs):
    train = []
    test = []
    iterations = []

    Test = Gradient_Descent(grote_matrix, test_data, labels, test_labels, lr,mt, False, False, False, False, False, bs)
    # Test.run_iteration()
    # print(Test.line_search_error(0.1))
    start = time.time()
    for i in range(10000):
        Test.run_iteration()
        #if i%1 == 0 and i > 1:
        if False:
            print(i)

            print("Train loss, test loss")
            # print(Test.calc_error_for_eval(False))
            # print(Test.calc_error_for_eval(True))
            print(Test.calc_classification_error(False))
            print(Test.calc_classification_error(True))
            train.append(Test.calc_error_for_eval(False))
            test.append(Test.calc_error_for_eval(True))
            iterations.append(i)
    end = time.time()
    print("Train class error, Test Class error")
    print("Learning rate: ", Test.learning_rate)
    print("Batch size: ", bs)
    print(Test.calc_classification_error(False))
    print(Test.calc_classification_error(True))
    setting = "Gradient descent"
    name = setting + "learning_rate-" + str(Test.learning_rate) + str(Test.momentum_term)
    #print(name)
    print("Time elapsed: ", end-start)
    # print(Test.learning_rate,Test.momentum_term)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # ax1.plot(iterations, train, color='black', label='Training Error')
    # ax1.plot(iterations, test, color='green', label='Test Error')
    # plt.title(setting + " - Time Elapsed: " + str(round(end - start, 2)))
    # plt.ylabel("Error")
    # plt.xlabel("Number of Iterations")
    #
    # plt.legend()
    # # plt.savefig(name)
    # plt.show()

if __name__ == '__main__':
    #cProfile.run('main()',sort='cumtime')
    lrs = [0.1,0.5,1]
    bs = [50, 200, 1000]
    for x in lrs:
        for y in bs:
            print(" ")
            main(x,0.1,y)
