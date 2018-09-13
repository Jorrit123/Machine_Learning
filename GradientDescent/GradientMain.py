from mnist import MNIST
from numpy.random import random
import math

mndata = MNIST('samples')

images, labels = mndata.load_training()

learning_rate = 1

index = 1
#print(mndata.display(images[index]))
#print(labels[index])

images3Or7, labels3Or7 = [], []
#select the 3s and 7s
for i in range(1, len(labels)):
    if labels[i] == 3 or labels[i] == 7:
        images[i].append(1)
        images3Or7.append(images[i])
        tempL = 0
        if labels[i] == 3:
            tempL = 0
        else:
            tempL = 1
        labels3Or7.append(tempL)

# all images are 784 pixels
# Normalize images
images3Or7Normalized=[]
for im in images3Or7:
    tempIm = []
    for p in im:
        tempIm.append(p/255)
    images3Or7Normalized.append(tempIm)

currentlyUsedList = images3Or7Normalized[1:50]

N = len(currentlyUsedList)

sameLength = 1
dim = len(images3Or7[0])
for im in images3Or7:
    if len(im) == dim and sameLength == 1:
        sameLength = 1
        dim = len(im)
    else:
        break

# Initialize weights
weights = []

for i in range(dim):
    weights.append(random())

print(weights)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Function that predicts the label of an image
def calc_prop(image):
    res = 0
    for d in range(dim):
        res += weights[d] * image[d]
    return sigmoid(res)

# Error function
def calc_error():
    res = 0
    for n in range(1, N):
        t_n = labels3Or7[n]
        y_n = calc_prop(currentlyUsedList[n])
        left_part = t_n * math.log(y_n)
        right_part = ((1-t_n)*math.log(1-y_n+0.000000001))
        res += left_part + right_part
    res = res / N
    return -res

# Partial derivative of the error function
def part_derivative(i):
    res = 0
    for n in range(1, N):
        t_n = labels3Or7[n]
        y_n = calc_prop(currentlyUsedList[n])
        res += (y_n - t_n)*currentlyUsedList[n][i]
    return res/N

# Run iterations
def run_iteration():
    for i in range(0, len(weights)-1):
        change = learning_rate * part_derivative(i)
        weights[i] -= change
    print("updated - Error:")
    print(calc_error())
    print(weights)

print("first error")
print(calc_error())
for x in range(0, 15):
    run_iteration()

