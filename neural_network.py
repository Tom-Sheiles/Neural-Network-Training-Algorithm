import sys
import math
import random
import numpy as np
import matplotlib as matplot


class Neurons:
    value = []
    weights = []
    net = []
    out = []
    error = []

    def __init__(self):
        self.weights = []
        self.error = []
        self.value = []
        self.net = []
        self.out = []

    def init_weights(self, n_weights):
        for i in range(n_weights):
            self.weights.append(random.uniform(0, 1))
            # self.weights.append(0.1)

    def init_value(self, value):
        self.value.append(value)

    def calculate_error(self, answer, step, batch):

        x = 0.5*((answer[batch][step] - self.out[batch])**2)
        self.error.append(x)

    def weighted_sum(self, step, input_neurons, bias, batch):

        total = 0
        for i in range(nInput):
            x = (input_neurons[i].weights[step] * input_neurons[i].value[batch])
            total += x
        total += bias
        self.net.append(total)

    def hidden_weighted_sum(self, step, hidden_neurons, bias, batch):

        total = 0
        for i in range(nHidden):
            x = (hidden_neurons[i].weights[step] * hidden_neurons[i].out[batch])
            total += x
        total += bias
        self.net.append(total)

    def sigmoid_function(self, batch):
        self.out.append(1/(1+math.exp(-self.net[batch])))


# TODO: Straight Up delete your function
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


def predict_answer(answers, batch):
    predictions = []
    for i in range(len(answers)):
        predictions.append(answers[i].out[batch])

    highest = predictions[0]
    highest_pos = 0

    for i in range(len(predictions)):
        if predictions[i] > highest:
            highest = predictions[i]
            highest_pos = i
    return highest_pos


def back_propagation(input_neurons, hidden_neurons, output_neurons, label, batch):

    gradient_vector = []
    totalErrors = []
    totalOutputs = []
    input_gradients = []

    # calculates the gradient values for the hidden weights
    # TODO: was len(hidden)
    for i in range(len(output_neurons)):
        error = -(label[i] - output_neurons[i].out[batch])
        output = (output_neurons[i].out[batch] * (1 - output_neurons[i].out[batch]))
        totalErrors.append(error)
        totalOutputs.append(output)
    for i in range(len(hidden_neurons)):
        for j in range(len(output_neurons)):
            gradient = (totalErrors[j] * totalOutputs[j] * hidden_neurons[i].out[batch])
            gradient_vector.append(gradient)

    # calculate value for error and output values for hidden and output neurons
    for i in range(len(totalErrors)):
        totalErrors[i] = totalErrors[i] * totalOutputs[i]
    for i in range(len(totalOutputs)):
        totalOutputs[i] = totalOutputs[i] * (1 - totalOutputs[i])

    x = 0
    # calculates the gradient values for the input weights
    for i in range(len(input_neurons)):
        for j in range(len(hidden_neurons)):
            error = 0
            # TODO: was hidden.weights
            for k in range(len(hidden_neurons[j].weights)):
                error += totalErrors[k] * hidden_neurons[j].weights[k]
            gradient = error * totalOutputs[x] * input_neurons[i].value[batch]
            x += 1
            if x >= len(totalOutputs):
                x = 0
            input_gradients.append(gradient)

    for i in range(len(gradient_vector)):
        input_gradients.append(gradient_vector[i])

    return input_gradients


def calculate_new_weights(gradient, current):

    # Calculate the final gradient error for all weights in the batch
    new_gradients = []
    for i in range(len(gradient[0])):
        weight_total = 0
        n = 0
        for j in range(len(gradient)):
            weight_total += gradient[j][i]
            n += 1
        weight_total /= n
        new_gradients.append(weight_total)

    for i in range(len(new_gradients)):
        new_gradients[i] = current[i] - (learningRate * new_gradients[i])

    return new_gradients


def update_weights(inputs, hidden, new):

    for i in range(len(inputs)):
        for j in range(len(inputs[i].weights)):
            inputs[i].weights[j] = new.pop(0)

    for i in range(len(hidden)):
        for j in range(len(hidden[i].weights)):
            hidden[i].weights[j] = new.pop(0)

    return


def get_current_weights(inputs, hidden):

    weights = []
    for i in inputs:
        for j in range(len(i.weights)):
            weights.append(i.weights[j])
    for i in hidden:
        for j in range(len(i.weights)):
            weights.append(i.weights[j])

    return weights


def train_neural_net(x, y):

    label = []
    gradient_vectors = []
    for i in range(len(trainLabel)):
        label.append(parse_label(trainLabel[i]))
    total_Error = []
    next_total_error = 0

    # init input neurons
    input_neurons = []
    for i in range(nInput):
        neuron = Neurons()
        neuron.init_weights(nHidden)
        input_neurons.append(neuron)

    # TODO: Remove this line and uncomment function later
    # input_neurons[1].weights[0] = 0.2

    # init hidden neurons
    hidden_neurons = []
    for i in range(nHidden):
        neuron = Neurons()
        neuron.init_weights(nOutPut)
        hidden_neurons.append(neuron)

    # TODO: Remove this line and uncomment function later
    # hidden_neurons[1].weights[1] = 0.2

    # init bias values for hidden layer and output layer
    hidden_bias = Neurons()
    output_bias = Neurons()
    hidden_bias.init_weights(nHidden)
    output_bias.init_weights(nOutPut)

    # init output neurons
    output_neurons = []
    for i in range(nOutPut):
        neuron = Neurons()
        output_neurons.append(neuron)

    initial_weigths = get_current_weights(input_neurons, hidden_neurons)
    print("\ninitial Weights: " + str(initial_weigths))

    predictions = []
    answers = []
    n = 0
    while True:
        # update weights in k batch size for epoch times
        for k in range(nEpochs):
            # calculate values for all inputs in batch
            for j in range(batchSize):
                # input values
                for i in range(nInput):
                    input_neurons[i].init_value(trainSet[j][i])

                # calculate the weighted sum of input values
                for i in range(nHidden):
                    hidden_neurons[i].weighted_sum(i, input_neurons, hidden_bias.weights[i], j)

                # calculate the sigmoid function of the hidden layer
                for i in range(nHidden):
                    hidden_neurons[i].sigmoid_function(j)

                # calculate the weighted sum of the output layer
                for i in range(nOutPut):
                    output_neurons[i].hidden_weighted_sum(i, hidden_neurons, output_bias.weights[i], j)

                # calculate the final output and error value for the initial values
                for i in range(nOutPut):
                    output_neurons[i].sigmoid_function(j)
                    output_neurons[i].calculate_error(label, i, j)
                    next_total_error += output_neurons[i].error[j]
                total_Error.append(next_total_error)

                # make prediction based on initial weights

                '''prediction = predict_answer(output_neurons, j)
                predictions.append(prediction)
                answers.append(trainLabel[j])'''

                '''print("Batch " + str(j))
                print("Predicted Answer: " + str(prediction))
                print("Actual Answer: " + str(trainLabel[j]))
                print("------------------")'''

                n += 1

                # calculate back propagation
                gradient_vectors.append(back_propagation(input_neurons, hidden_neurons, output_neurons, label[j], j))
            current_weights = get_current_weights(input_neurons, hidden_neurons)
            new_weights = calculate_new_weights(gradient_vectors, current_weights)
            update_weights(input_neurons, hidden_neurons, new_weights)

            print("Epoch " + str(k + 1))

            total_Error.clear()
            gradient_vectors.clear()
            for i in range(len(input_neurons)):
                input_neurons[i].value.clear()
            for i in range(len(hidden_neurons)):
                hidden_neurons[i].net.clear()
                hidden_neurons[i].out.clear()
            for i in range(len(output_neurons)):
                output_neurons[i].error.clear()
                output_neurons[i].net.clear()
                output_neurons[i].out.clear()
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == answers[i]:
                correct += 1

        print("\nTraining Complete:")
        print(n)
        print("Initl Weights: " + str(initial_weigths))
        print("Final Weights: " + str(get_current_weights(input_neurons, hidden_neurons)))
        print("\nAccuracy: " + str(correct/n))

        continueInput = input("Continue Training? (y/n): ")
        if continueInput == "n":
            break


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
batchSize = 2
learningRate = .3

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
