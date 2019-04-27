import sys
import math
import random
import numpy as np
import matplotlib as matplot


def weighted_sum(values, weights):

    weight_sum = 0
    for i in range(len(weights)):
        weight_sum += (values[i] * weights[i])

    print(weight_sum)

    return


def init_neurons():

    neurons = []
    for i in range(nInput):
        for j in range(2):
            neurons.append(random.uniform(0, 1))

    for i in range(nHidden):
        for j in range(2):
            neurons.append(random.uniform(0, 1))

    print("\nNumber of neurons: " + str(len(neurons)))
    return neurons


def train_neural_net():
    neuron_weights = init_neurons()
    print(neuron_weights)

    weighted_sum(trainSet[0], neuron_weights)


argumentNumber = len(sys.argv)
print(str(len(sys.argv) - 1) + " Arguments Entered: ")

if argumentNumber == 8:
    nInput = int(sys.argv[1])
    nHidden = int(sys.argv[2])
    nOutPut = int(sys.argv[3])
    trainSet = sys.argv[4]
    trainLabel = sys.argv[5]
    testSet = sys.argv[6]
    testLabel = sys.argv[7]
else:
    print("Not all arguments input. (nInput, nHidden, nOutput, Training Set, Training labels, Test Set, Test labels)")
    exit(1)

nEpochs = 30
batchSize = 20
learningRate = 3

for i in range(1, len(sys.argv)):
    print(sys.argv[i])

# Load training data
print("\nLoading Training Set... ")
trainSet = np.loadtxt(trainSet, float, delimiter=",")
print("Training Set Loaded. ")

print("\nLoading Training Labels... ")
trainLabel = np.loadtxt(trainLabel, int, delimiter=",")
print("Training Labels Loaded. ")

train_neural_net()
