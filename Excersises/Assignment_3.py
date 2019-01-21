import numpy as np
import matplotlib.pyplot as plt


train_data = np.loadtxt('lasso_data\data1_input_train').T
train_data = train_data/np.linalg.norm(train_data,axis=0)
bias = np.ones(train_data[:,0].size)

train_data = np.concatenate([train_data,bias[:,None]],axis=1)
train_labels = np.loadtxt('lasso_data\data1_output_train')
test_data = np.loadtxt('lasso_data\data1_input_val').T
bias = np.zeros(test_data[:,0].size)
test_data = np.concatenate([test_data,bias[:,None]], axis=1)
test_labels = np.loadtxt('lasso_data\data1_output_val')

# np.random.seed(1)


#Generating corrolated data
n = 3
p = 10 # train data size
#w = [2, 3, 0] # example 1a
w = [-2, 3, 0] # example 1b
sigma = 1
x=np.zeros((n,p))
x[:2,:]=np.random.normal(0,1,(2,p))
x[2,:]=2 / 3 * x[1,:]+2 / 3 * x[2,:]+1 / 3 * np.random.normal(size=p);

y = np.dot(w,x) + np.random.normal(size=p);
# print(y)
corrolated = True
if corrolated:
    train_data = x.T/np.linalg.norm(x.T,axis=0)
    train_labels = y

#The lasso class

class Lasso():
    def __init__(self,train_data,train_labels,test_data,test_labels,gamma):
        self.train_data = train_data
        self.train_labels = train_labels
        self.gamma = gamma
        self.N = train_data[:,0].size
        self.d = train_data[0,:].size
        self.weights = np.ones(self.d)

    def predict(self):
        return np.dot(self.train_data, self.weights)

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
        for j in range(self.d):
            x_j = self.train_data[:, j]
            z = np.dot(x_j, self.train_labels - self.predict() + self.weights[j] * x_j)
            self.weights[j] = self.soft_thresholding_operator(z, self.gamma)

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
            self.weights = np.ones(self.d)
            self.train_data = [x for j,x in enumerate(data_splits) if j!=i]
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data/np.linalg.norm(self.train_data,axis=0)
            self.train_labels = [x for j, x in enumerate(label_splits) if j != i]
            self.train_labels = np.concatenate(self.train_labels)
            for k in range(10):
                self.iteration()
            self.train_data = data_splits[i]
            self.train_labels = label_splits[i]
            total_error += self.error()

        return total_error

if __name__ == '__main__':

    cross_validating = False
    if not cross_validating:
        weights = []
        gammas = np.logspace(-5,2,100)
        for i in gammas:
            LassoObj = Lasso(train_data,train_labels,test_data,test_labels,i)
            for iter in range(100):
                LassoObj.iteration()
            weights.append(LassoObj.weights.flatten())
        weights = np.array(weights).T

        for weight in weights:
            plt.plot(gammas,weight)
        plt.xscale('log')
        plt.xlabel("gamma")
        plt.show()


    else:
        gamma = []
        errors = []
        gammas = np.logspace(-5, 1, 100)
        for g in gammas:
            LassoStep = Lasso(train_data, train_labels, test_data, test_labels, g)
            error = LassoStep.cross_validate(5)
            if error < 100:
                gamma.append(g)
                errors.append(error)
            # print(g, ": ", error)
        index = np.argmin(np.array(errors))
        print(gamma[index],index)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(gamma, errors, color='black', label='gamma')
        plt.xscale('log')
        plt.legend()
        plt.show()