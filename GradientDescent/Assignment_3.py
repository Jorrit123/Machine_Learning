import numpy as np


train_data = np.loadtxt('lasso_data\data1_input_train')
train_labels = np.loadtxt('lasso_data\data1_output_train')
test_data = np.loadtxt('lasso_data\data1_input_val')
test_labels = np.loadtxt('lasso_data\data1_output_val')
print(train_data.shape)


class Lasso():
    def __init__(self,train_data,train_labels,test_data,test_labels,gamma):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.gamma = gamma
        self.N = train_data[:,0].size
        self.d = train_data[0,:].size
        self.chi = np.zeros((self.N,self.N))
        self.weights = np.zeros(self.d)
        self.b = self.d
        self.predictions = np.zeros(self.N)

    def calculate_stuff(self):
        self.chi = np.dot(self.train_data,self.train_data.T)/self.N
        self.b = np.dot(self.train_data,self.train_labels)

    def calculate_weights(self):
        self.weights = np.dot(np.linalg.inv(self.chi),(self.b-self.gamma))

    def predict(self):
        self.calculate_stuff()
        self.calculate_weights()
        self.predictions = np.dot(self.weights,self.train_data)



if __name__ == '__main__':
    Lasso = Lasso(train_data,train_labels,test_data,test_labels,0)

    Lasso.predict()
    print(Lasso.predictions)
