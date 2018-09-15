import sys

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


#start cropping
dim = len(images3Or7[0])

usedPixels = []
for i in range(0, dim):
    usedPixels.append(0)


for x in range(0, dim):
    for im in images3Or7Normalized:
        usedPixels[x] += im[x]


usedDim = 0
for p in usedPixels:
    if p>0:
        usedDim += 1

croppedImages = []
for im in images3Or7Normalized:
    tempIm = []
    for i in range(0, dim):
        if usedPixels[i] > 0:
            tempIm.append(im[i])
    croppedImages.append(tempIm)

#end cropping

currentlyUsedList = croppedImages

N = len(currentlyUsedList)

#test same dimension
# sameLength = 1
# dim = len(images3Or7[0])
# for im in images3Or7:
#     if len(im) == dim and sameLength == 1:
#         sameLength = 1
#         dim = len(im)
#     else:
#         break



# Initialize weights
weights = []

for i in range(len(currentlyUsedList[0])):
    weights.append(random())

print(weights)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Function that predicts the label of an image
def calc_prop(image):
    res = 0
    for d in range(len(image)):
        res += weights[d] * image[d]
    return sigmoid(res)


props = []
for x in range(N):
    props.append(0)

def calc_prop2(index):
    return props[index]


# Error function
def calc_error():
    res = 0
    for n in range(1, N):
        t_n = labels3Or7[n]
        y_n = calc_prop(currentlyUsedList[n])
        left_part = t_n * math.log(y_n)
        right_part = ((1-t_n)*math.log(1-y_n+sys.float_info.min))
        res += left_part + right_part
    res = res / N
    return -res

# Partial derivative of the error function
def part_derivative(i):
    res = 0
    for n in range(1, N):
        t_n = labels3Or7[n]
        # y_n = calc_prop(currentlyUsedList[n])
        y_n = calc_prop2(n)
        res += (y_n - t_n)*currentlyUsedList[n][i]
    return res/N

# Run iterations
def run_iteration():
    for i in range(N):
        props[i] = calc_prop(currentlyUsedList[i])
    for i in range(0, len(weights)):
        change = learning_rate * part_derivative(i)
        weights[i] -= change
    print("updated - Error:")
    print(calc_error())
    # print(weights)

print("first error")
print(calc_error())
while True:
    run_iteration()

