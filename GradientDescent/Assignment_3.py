import numpy as np


train_data = np.loadtxt('lasso_data\data1_input_train').T
bias = np.zeros(train_data[:,0].size)
print(train_data.shape,bias.shape)
train_data = np.concatenate([train_data,bias[:,None]],axis=1)
train_labels = np.loadtxt('lasso_data\data1_output_train')
test_data = np.loadtxt('lasso_data\data1_input_val').T
bias = np.zeros(test_data[:,0].size)
test_data = np.concatenate([test_data,bias[:,None]], axis=1)
test_labels = np.loadtxt('lasso_data\data1_output_val')



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
        self.weights = np.random.normal(size=self.d)
        self.weights = np.zeros(self.d)
        self.b = np.random.normal(size=self.d)
        self.predictions = np.zeros(self.N)


    # def calculate_chi_b(self):
    #     self.chi = np.dot(self.train_data.T,self.train_data)/self.N
    #     self.b = np.dot(self.train_labels,self.train_data)
    #
    # def calculate_weights(self):
    #     self.weights = np.dot(self.b-self.gamma,np.linalg.inv(self.chi))

    def predict(self):
        self.predictions = np.dot(self.train_data, self.weights)
        return self.predictions

    def update_weight(self,j):
        temp =self.weights[j]*self.train_data[:,j]
        predictions_minus_j = self.predict()-temp
        predict_temp = self.train_labels - predictions_minus_j
        z = (1/self.N)*np.sum(train_data[:,j]*predict_temp)
        self.weights[j] = self.soft_thresholding_operator(z,self.gamma)

    @staticmethod
    def soft_thresholding_operator(z,y):
        if z > 0 and y < np.abs(z):
            return z - y
        if z < 0 and y < np.abs(z):
            return z + y
        else:
            return 0

    def iteration(self):
        for j in range(self.weights.size):
            self.update_weight(j)

    def error(self):
        return np.sum((self.predict()-self.train_labels)**2)




if __name__ == '__main__':
    Lasso = Lasso(train_data,train_labels,test_data,test_labels,1)
    for i in range(200):
        Lasso.iteration()
        print(Lasso.error())
        # print(Lasso.weights)


