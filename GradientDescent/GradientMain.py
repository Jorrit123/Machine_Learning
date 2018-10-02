from scipy.optimize import fmin

def square(x):
    return -x**2

print(fmin(square,500))



