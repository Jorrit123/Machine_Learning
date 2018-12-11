import numpy as np
import matplotlib.pyplot as plt


x0 = 0
v = 0.01


def dx(u,dt):
    noise = np.random.normal(0,np.sqrt(v))
    return u*dt+noise


def u(t,x,T):
    inner = x/(v*(T-t))
    outer = np.tanh(inner)-x
    return outer/(T-t)


size = 1000
x_grid = []
x = np.zeros(size)
dt = 0.01
T = size*dt
Time = np.arange(0, T, dt)

for g_number in range (15):
    for counter in range(size-1 ):
        u_temp = u(Time[counter],x[counter],T)
        x[counter+1] = dx(u_temp,dt)+x[counter]
    x_grid.append(x)
    x = np.zeros(size)

for plot in x_grid:
    plt.plot(plot)
plt.plot(Time,x)
plt.show()
#print(x)