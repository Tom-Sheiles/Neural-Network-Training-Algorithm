import sys
import math
import random
import numpy as np
import matplotlib as matplot


class Neurons:
    value = float()
    weights = []
    net = float()
    out = float()
    error = float()
    totalError = float()

    def __init__(self):
        self.weights = []
        self.error = 0
        self.value = 0
        self.net = 0
        self.out = 0
        self.totalError = 0

    def init_weights(self, n_weights):
        for i in range(n_weights):
            self.weights.append(random.uniform(0, 1))

    def init_value(self, value):
        self.value = value

    def calculate_error(self, answer, step):

        x = 0.5*((answer[step] - self.out)**2)
        self.error = x

    def weighted_sum(self, step, input_neurons, bias):

        total = 0
        for i in range(nInput):
            x = (input_neurons[i].weights[step] * input_neurons[i].value)
            total += x
        total += bias
        self.net = total
        print()

    def hidden_weighted_sum(self, step, hidden_neurons, bias):

        total = 0
        for i in range(nHidden):
            x = (hidden_neurons[i].weights[step] * hidden_neurons[i].out)
            total += x
        total += bias
        self.net = total
        print()

    def sigmoid_function(self):
        self.out = (1/(1+math.exp(-self.net)))


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


def parse_label(labels):

    label = [0] * nOutPut
    pos = 0
    for i in range(labels):
        pos += 1
    label[pos] = 1
    return label


def train_neural_net(x, y):

    label = parse_label(trainLabel[0])
    total_Error = 0

    # init input neurons
    input_neurons = []
    for i in range(nInput):
        neuron = Neurons()
        neuron.init_weights(nHidden)
        neuron.init_value(trainSet[0][i])
        input_neurons.append(neuron)

    # init hidden neurons
    hidden_neurons = []
    for i in range(nHidden):
        neuron = Neurons()
        neuron.init_weights(nOutPut)
        hidden_neurons.append(neuron)

    # init bias values for hidden layer and output layer
    hidden_bias = Neurons()
    output_bias = Neurons()
    hidden_bias.init_weights(1)
    output_bias.init_weights(1)

    # init output neurons
    output_neurons = []
    for i in range(nOutPut):
        neuron = Neurons()
        output_neurons.append(neuron)

    # calculate the weighted sum of input values
    for i in range(nHidden):
        hidden_neurons[i].weighted_sum(i, input_neurons, hidden_bias.weights[0])

    # calculate the sigmoid function of the hidden layer
    for i in range(nHidden):
        hidden_neurons[i].sigmoid_function()

    # calculate the weighted sum of the output layer
    for i in range(nOutPut):
        output_neurons[i].hidden_weighted_sum(i, hidden_neurons, output_bias.weights[0])

    # calculate the final output and error value for the initial values
    for i in range(nOutPut):
        output_neurons[i].sigmoid_function()
        output_neurons[i].calculate_error(label, i)
        total_Error += output_neurons[i].error


    print()


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

train_neural_net(trainSet, trainLabel)
