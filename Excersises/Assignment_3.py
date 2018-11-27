import numpy as np
import matplotlib.pyplot as plt


train_data = np.loadtxt('lasso_data\data1_input_train').T
bias = np.zeros(train_data[:,0].size)
print(train_data.shape,bias.shape)
train_data = np.concatenate([train_data,bias[:,None]],axis=1)
train_labels = np.loadtxt('lasso_data\data1_output_train')
test_data = np.loadtxt('lasso_data\data1_input_val').T
bias = np.zeros(test_data[:,0].size)
test_data = np.concatenate([test_data,bias[:,None]], axis=1)
test_labels = np.loadtxt('lasso_data\data1_output_val')

np.random.seed(1)


#Generating corrolated date
n = 3
p = 1000 # train data size
w = [2, 3, 0] # example 1a
# w = [-2, 3, 0] # example 1b
sigma = 1
x=np.zeros((3,p))
x[:2,:]=np.random.normal(0,1,(2,p))
x[2,:]=2 / 3 * x[1,:]+2 / 3 * x[2,:]+1 / 3 * np.random.normal(size=p);
y = np.dot(w,x) + np.random.normal(size=p);


#The lasso class

class Lasso():
    def __init__(self,train_data,train_labels,test_data,test_labels,lam,alpha):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.gamma = lam * alpha
        self.lam = lam
        self.alpha = alpha
        self.N = train_data[:,0].size
        self.d = train_data[0,:].size
        self.chi = np.zeros((self.N,self.N))
        self.weights = np.random.normal(size=self.d)
        self.b = np.random.normal(size=self.d)
        self.predictions = np.zeros(self.N)


    # def calculate_chi_b(self):
    #     self.chi = np.dot(self.train_data.T,self.train_data)/self.N
    #     self.b = np.dot(self.train_labels,self.train_data)
    #
    # def calculate_weights(self):
    #     self.weights = np.dot(self.b-self.gamma,np.linalg.inv(self.chi))

    def set_alpha(self,value):
        self.alpha = value

    def predict(self):
        self.predictions = np.dot(self.train_data, self.weights)
        return self.predictions

    def update_weight(self,j):
        temp =self.weights[j]*self.train_data[:,j]
        predictions_minus_j = self.predict()-temp
        predict_temp = self.train_labels - predictions_minus_j
        z = (1/self.N)*np.sum(self.train_data[:,j]*predict_temp)
        threshold = self.soft_thresholding_operator(z,self.gamma)
        self.weights[j] = threshold/(1+self.lam*(1-self.alpha))

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

    def cross_validate(self,k):
        total_error = 0

        data_splits = []
        label_splits = []
        size = int(self.N / k)
        for i in range(k):
            start = i*size
            data_splits.append(self.train_data[start:start+size,:])
            label_splits.append(self.train_labels[start:start+size])

        for i in range(k):
            self.weights = np.random.normal(size=self.d)
            self.train_data = [x for j,x in enumerate(data_splits) if j!=i]
            self.train_data = np.concatenate(self.train_data)
            self.train_labels = [x for j, x in enumerate(label_splits) if j != i]
            self.train_labels = np.concatenate(self.train_labels)
            for k in range(20):
                self.iteration()
            self.train_data = data_splits[i]
            self.train_labels = label_splits[i]
            total_error += self.error()

        return total_error

if __name__ == '__main__':
    # Lasso = Lasso(train_data,train_labels,test_data,test_labels,0.3,0.3)
    # print(Lasso.cross_validate(5))
    # for i in range(200):
    #     Lasso.iteration()
    #     print(Lasso.error())
    #     # print(Lasso.weights)


    gamma = []
    errors = []
    gamma_values = np.arange(0.15,0.6,0.05)
    for g in gamma_values:
        LassoStep = Lasso(train_data, train_labels, test_data, test_labels, g, g)
        gamma.append(g)
        error = LassoStep.cross_validate(5)
        errors.append(error)
        print(g, ": ", error)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(gamma, errors, color='black', label='gamma')

    plt.legend()
    plt.show()