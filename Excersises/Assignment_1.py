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


test_images, test_labels = mndata.load_testing()
test_images = np.array(test_images)/255
test_bias = np.ones(test_images[:,0].shape)
test_images = np.concatenate([test_images,test_bias[:,None]], axis=1)
test_labels = np.array(test_labels)
test_data = np.concatenate([test_images, test_labels[:,None]], axis=1)
test_data = test_data[(test_data[:,-1] == 3) | (test_data[:,-1] == 7)]
test_data[:,-1][test_data[:,-1]==3]=0
test_data[:,-1][test_data[:,-1]==7]=1

data = data[:test_data[:,0].size]

print(data.shape)


#labels = labels[labels == 3 or labels ==7]

class Gradient_Descent():
    temp_gradients =[]

    def __init__(self,data, test_data,momentum=False,decay=False,newton=False,line_search=False,conjugate=False, batch_size = 0):
        self.pixels = 785
        self.test_data = test_data
        self.data = data
        self.weights = np.random.normal(scale=1,size=self.pixels)
        self.batch_size = self.data[:, 1].size
        self.batch_size = batch_size if batch_size != 0 else self.batch_size
        self.probabilities = np.zeros(self.batch_size)
        self.learning_rate = 0.5
        self.momentum_term = 0.6
        self.weight_decay_rate = 0.1
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
        self.probabilities = 1 / (1 + np.exp(-np.dot(prop_data[self.batch_indices, :-1], x)))
        return self.probabilities

    def calc_error(self,probs):
        error = self.data[:,-1]*np.log(probs) + (1-self.data[:,-1])*np.log(1-probs)
        if self.decay:
            error += (self.weight_decay_rate / (2 * self.batch_size)) * np.sum(self.weights**2)
        return (-1 / self.batch_size) * np.sum(error)

    def calc_error_for_eval(self, is_test = False):
        error_data = self.data
        if is_test:
            error_data = self.test_data
        self.batch_indices = np.arange(error_data[:,1].size)
        probs = self.calc_prop(is_test = is_test)
        error = error_data[:, -1] * np.log(probs) + (1 - error_data[:, -1]) * np.log(1 - probs)
        # if self.decay:
        #     error += (self.weight_decay_rate / (2 * self.batch_size)) * np.sum(self.weights**2)
        return (-1 / error_data[:,1].size) * np.sum(error)

    def gradient(self):
        gradients = self.division * np.dot((self.probabilities - self.data[self.batch_indices, -1]), self.data[self.batch_indices, :-1])
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
        hessian = np.dot(np.dot(self.probabilities*(1-self.probabilities),self.data[:,:-1]),self.data[:,:-1].T)
        return np.linalg.inv(self.division * hessian + np.identity(self.pixels) * (self.weight_decay_rate / self.batch_size))

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
        self.batch_indices = np.arange(class_data[:, 1].size)
        self.probabilities = self.calc_prop(is_test=is_test)
        guess = np.round(self.probabilities)
        guess = list(map(bool, guess))
        labels = list(map(bool,class_data[:,-1]))
        right_guess = np.array(guess) == np.array(labels)
        return 1-np.array(right_guess).sum()/class_data[:,1].size

    def prepare_iteration(self):
        self.batch_indices = np.arange(self.data[:, 1].size)

def main():
    train = []
    test = []
    iterations = []
    if __name__ == '__main__':
        Test = Gradient_Descent(data, test_data, False, False, True, False, False, 0)
        # Test.run_iteration()
        # print(Test.line_search_error(0.1))
        start = time.time()
        for i in range(50):
            Test.run_iteration()
            #if i%100 == 0 and i > 1:
            if True:
                print(i)

                print("Train loss, test loss")
                print(Test.calc_error_for_eval(False))
                print(Test.calc_error_for_eval(True))
                train.append(Test.calc_error_for_eval(False))
                test.append(Test.calc_error_for_eval(True))
                iterations.append(i)
        end = time.time()
        print("Train class error, Test Class errror")
        print(Test.calc_classification_error(False))
        print(Test.calc_classification_error(True))
        setting = "Newton"
        name = setting + "learning_rate-" + str(Test.learning_rate) + str(Test.momentum_term)
        print(name)
        # print(Test.learning_rate,Test.momentum_term)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(iterations, train, color='black', label='Training Error')
    ax1.plot(iterations, test, color='green', label='Test Error')
    plt.title(setting + " - Time Elapsed: " + str(round(end - start, 2)))
    plt.ylabel("Error")
    plt.xlabel("Number of Iterations")

    plt.legend()
    # plt.savefig(name)
    plt.show()

if __name__ == '__main__':
    #cProfile.run('main()',sort='cumtime')
    main()