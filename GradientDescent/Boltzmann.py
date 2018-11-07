import numpy as np
import matplotlib.pyplot as plt


class Boltzmann_machine():
    def __init__(self,P,N,learning_rate):
        self.P = P
        self.N = N
        self.eta = learning_rate

        self.Data = np.random.choice([-1, 1], size=(N,P))

        self.clamped_means = np.zeros(shape = 10)
        self.clamped_correlations = np.zeros(shape = (10,10))

        self.means = np.zeros(shape=10)
        self.correlations = np.zeros(shape=(10, 10))

        self.states = np.random.choice([-1,1], size = N)

        self.weights = np.random.normal(size=(N,N))
        self.thetas = np.random.normal(size = N)

        self.clamped_means = np.sum(self.Data,axis=1)*1/self.P

        self.clamped_correlations = np.dot(self.Data,self.Data.T)*1/self.P

    @staticmethod
    def sigmoid(x):
        return (np.exp(x)) / (np.exp(x) + 1)

    def sequential_dynamics(self):
        states = np.zeros(10)
        correlation = np.zeros((10,10))
        for i in range(500):
            n = np.random.choice(range(self.N))
            h = np.dot(self.weights[n,:],self.states) - self.weights[n,n]*self.states[n]
            probability = self.sigmoid(-self.states[n]*(h + self.thetas[n]))
            if probability > np.random.uniform(0,1):
                self.states[n] = -self.states[n]
            states += self.states
            correlation += np.outer(self.states,self.states)

        self.means = states/500
        print(self.means)
        print(self.clamped_means)
        self.correlations = correlation/500

    def update_weights(self):
        self.sequential_dynamics()
        self.thetas += self.eta*(self.clamped_means-self.means)
        self.weights += self.eta*(self.clamped_correlations-self.correlations)


if __name__ == '__main__':
    Boltzmann_machine = Boltzmann_machine(200,10,1)
    change_in_weights = []
    change_in_thetas = []
    for i in range(200):
        Boltzmann_machine.update_weights()
        change_in_weights.append(np.mean(np.abs(Boltzmann_machine.eta*(Boltzmann_machine.clamped_correlations-Boltzmann_machine.correlations))))
        change_in_thetas.append(np.mean(np.abs((Boltzmann_machine.eta*(Boltzmann_machine.clamped_means-Boltzmann_machine.means)))))

    iterations = np.arange(200)
    change_in_weights = np.array(change_in_weights)
    change_in_thetas = np.array(change_in_thetas)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(iterations, change_in_weights, color='black', label='change in weights')
    ax1.plot(iterations, change_in_thetas, color='green', label='change in thetas')

    plt.legend()
    plt.show()




    #print(Boltzmann_machine.means, Boltzmann_machine.correlations)



