import numpy as np
import matplotlib.pyplot as plt


train_data = np.loadtxt('lasso_data\data1_input_train').T
train_data = train_data[:,1:]
bias = np.zeros(train_data[:,0].size)

train_data = np.concatenate([train_data,bias[:,None]],axis=1)
train_labels = np.loadtxt('lasso_data\data1_output_train')
test_data = np.loadtxt('lasso_data\data1_input_val').T
bias = np.zeros(test_data[:,0].size)
test_data = np.concatenate([test_data,bias[:,None]], axis=1)
test_labels = np.loadtxt('lasso_data\data1_output_val')

#np.random.seed(1)


#Generating corrolated data
n = 3
p = 1000 # train data size
w = [2, 3, 0] # example 1a
# w = [-2, 3, 0] # example 1b
sigma = 1
x=np.zeros((n,p))
x[:2,:]=np.random.normal(0,1,(2,p))
x[2,:]=2 / 3 * x[1,:]+2 / 3 * x[2,:]+1 / 3 * np.random.normal(size=p);

y = np.dot(w,x) + np.random.normal(size=p);
# print(y)
corrolated = False
if corrolated:
    train_data = x.T
    train_labels = y

#The lasso class

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
        self.b = np.random.normal(size=self.d)
        self.predictions = np.zeros(self.N)


    # def calculate_chi_b(self):
    #     self.chi = np.dot(self.train_data.T,self.train_data)/self.N
    #     self.b = np.dot(self.train_labels,self.train_data)
    #
    # def calculate_weights(self):
    #     self.weights = np.dot(self.b-self.gamma,np.linalg.inv(self.chi))

    # def set_alpha(self,value):
    #     self.alpha = value

    def predict(self):
        self.predictions = np.dot(self.train_data, self.weights)
        return self.predictions

    def update_weight(self,j):
        temp =self.weights[j]*self.train_data[:,j]
        test = self.predict()
        predictions_minus_j = self.predict()-temp
        error = self.train_labels - predictions_minus_j
        z = (1/self.N)*np.sum(self.train_data[:,j]*error)
        threshold = self.soft_thresholding_operator(z,self.gamma)
        self.weights[j] = threshold

    @staticmethod
    def soft_thresholding_operator(z,y):
        if y >= np.abs(z):
            return 0
        else:
            if z > 0:
                return z - y
            else:
                return z + y

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

    cross_validating = False
    if not cross_validating:
        LassoObj = Lasso(train_data,train_labels,test_data,test_labels,0.1)
        # print(Lasso.cross_validate(5))
        iter = 20
        iterations = np.zeros(iter+1)
        iterations[0] = 0
        b_1 = np.zeros(iter+1)
        b_2 = np.zeros(iter+1)
        b_3 = np.zeros(iter+1)
        b_1[0] = LassoObj.weights[0]
        b_2[0] = LassoObj.weights[1]
        b_3[0] = LassoObj.weights[2]

        for i in range(iter):
            LassoObj.iteration()
            print(LassoObj.error())
            iterations[i+1]=i+1
            b_1[i+1] = LassoObj.weights[0]
            b_2[i+1] = LassoObj.weights[1]
            b_3[i+1] = LassoObj.weights[2]
            # print(Lasso.weights)
        print(LassoObj.weights)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # axl_2 = fig.add_subplot(111)
        # axl_3 = fig.add_subplot(111)
        # axl_4 = fig.add_subplot(111)


        ax1.plot(iterations, b_1, color='black', label='b1')
        ax1.plot(iterations, b_2, color='blue', label='b2')
        ax1.plot(iterations, b_3, color='green', label='b3')

        plt.legend()
        plt.title('First Coordinate')
        plt.show()

    else:
        gamma = []
        errors = []
        gamma_values = np.arange(0.05,0.2,0.001)
        for g in gamma_values:
            LassoStep = Lasso(train_data, train_labels, test_data, test_labels, g)
            gamma.append(g)
            error = LassoStep.cross_validate(5)
            errors.append(error)
            print(g, ": ", error)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(gamma, errors, color='black', label='gamma')

        plt.legend()
        plt.show()