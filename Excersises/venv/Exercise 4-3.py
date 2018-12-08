import numpy as np
import matplotlib.pyplot as plt


x0 = 0
v = 1


def dx(u,dt):
    noise = np.random.normal(0,np.sqrt(1))
    return u*dt+noise


def u(t,x):
    inner = x/(v)
    outer = np.tanh(inner-x)
    return outer/(T-t)


size = 100
x = np.zeros(size)
dt = 0.1
T = 100*dt
Time = np.arange(0, T, dt)

for counter in range(1,size):
    u_temp = u(Time[counter],x[counter-1])
    dt_temp = dx(u_temp,dt)
    x[counter] = x[counter-1] + dt_temp

plt.plot(Time,x)
plt.show()
print(x)