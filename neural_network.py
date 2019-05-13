import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as deepcopy
import time


# Class defines the attributes and methods for each individual neuron in the network
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
            self.weights.append(0.01 * random.uniform(0, 1))

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
        try:
            self.out.append(1/(1+math.exp(-self.net[batch])))
        except:
            self.net[batch] *= 0.01
            self.out.append(1 / (1 + math.exp(-self.net[batch])))


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


# Calculates the gradient descent values for the network weights
def back_propagation(input_neurons, hidden_neurons, output_neurons, label, batch):

    gradient_vector = []
    totalErrors = []
    totalOutputs = []
    input_gradients = []

    # calculates the gradient values for the hidden weights
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
            x = gradient[j][i]
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


def save_weights(input, hidden, hidden_bias, output_bias):

    f = open("SavedWeights.txt", "a")
    for i in range(len(input)):
        for j in range(len(input[i].weights)):
            f.write(str(input[i].weights[j]) + "\n")
    f.write("\n")
    for i in range(len(hidden)):
        for j in range(len(hidden[i].weights)):
            f.write(str(hidden[i].weights[j]) + "\n")
    f.write("\n")

    for i in range(len(hidden_bias.weights)):
        f.write(str(hidden_bias.weights[i]) + "\n")
    f.write("\n")

    for i in range(len(output_bias.weights)):
        f.write(str(output_bias.weights[i]) + "\n")
    f.write("\n")

    f.close()
    return


def quadratic_cost_function(batch_size, output, labels, batch):

    labels_vector = []
    for i in range(len(labels)):
        labels_vector.append(parse_label(labels[i]))

    output_vector = []
    for i in range(len(output)):
        output_vector.append(output[i].out[batch])
        pass

    total = 0
    for i in range(len(output_vector)):
        total += (abs(output_vector[i] - labels_vector[batch][i]))
    total = total ** 2
    total *= 1/(2*batch_size)

    return total


def test_accuracy(inputs, hidden, output, hidden_bias, output_bias):

    test_subset = 1000
    predictions = []
    answers = []
    correct = 0
    for step in range(test_subset):

        for i in range(nInput):
            inputs[i].init_value(testSet[step][i])

        # calculate the weighted sum of input values
        for i in range(nHidden):
            hidden[i].weighted_sum(i, inputs, hidden_bias.weights[i], step)
        # calculate the sigmoid function of the hidden layer
        for i in range(nHidden):
            hidden[i].sigmoid_function(step)

        # calculate the weighted sum of the output layer
        for i in range(nOutPut):
            output[i].hidden_weighted_sum(i, hidden, output_bias.weights[i], step)

        # calculate the final output and error value for the initial values
        for i in range(nOutPut):
            output[i].sigmoid_function(step)

        prediction = predict_answer(output, step)
        predictions.append(prediction)
        answers.append(testLabel[step])

    for i in range(len(predictions)):
        if predictions[i] == answers[i]:
            correct += 1

    accuracy = correct/test_subset
    print("Test Accuracy: " + str(accuracy))

    return accuracy


def train_neural_net():

    start = time.clock()
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

    # init hidden neurons
    hidden_neurons = []
    for i in range(nHidden):
        neuron = Neurons()
        neuron.init_weights(nOutPut)
        hidden_neurons.append(neuron)

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
    print("\nBeginning Training:")

    predictions = []
    answers = []
    xs = []
    ys = []
    accuracy = []
    batch_n = -1
    n = 0
    last_answer_size = 0
    while True:
        # update weights in k batch size for epoch times
        for k in range(nEpochs):
            # calculate values for all inputs in batch
            p = 0
            for z in range(len(trainSet)):
                for j in range(batchSize):

                    batch_n += 1
                    if batch_n >= len(trainSet):
                        batch_n = 0

                    # input values
                    for i in range(nInput):
                        input_neurons[i].init_value(trainSet[batch_n][i])

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
                    prediction = predict_answer(output_neurons, j)
                    predictions.append(prediction)
                    answers.append(trainLabel[batch_n])

                    n += 1

                    # calculate back propagation
                    gradient_vectors.append(back_propagation(input_neurons, hidden_neurons, output_neurons, label[j], j))

                current_weights = get_current_weights(input_neurons, hidden_neurons)
                new_weights = calculate_new_weights(gradient_vectors, current_weights)
                update_weights(input_neurons, hidden_neurons, new_weights)

                len_answers = len(answers)
                sub_set_answers = answers[last_answer_size:len_answers]
                for n_answers in range(len(sub_set_answers)):
                    quadratic_cost = quadratic_cost_function(batchSize, output_neurons, sub_set_answers, n_answers)
                    xs.append(quadratic_cost)
                last_answer_size += batchSize

                # reset values for neurons
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

            print("Epoch " + str(k + 1))
            correct = 0
            for i in range(len(predictions)):
                if predictions[i] == answers[i]:
                    correct += 1
            print("Training Accuracy: " + str(correct / len(answers)))
            accuracy.append(correct/len(answers))

            test_set_accuracy = test_accuracy(deepcopy(input_neurons), deepcopy(hidden_neurons),
                                              deepcopy(output_neurons), hidden_bias, output_bias)
            ys.append(test_set_accuracy)

        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == answers[i]:
                correct += 1

        print("\nTraining Complete:")
        print("Initl Weights: " + str(initial_weigths))
        print("Final Weights: " + str(get_current_weights(input_neurons, hidden_neurons)))
        print("Training Accuracy: " + str(correct/len(answers)))
        print("Test Accuracy: " + str(ys[-1]))
        end = time.clock()
        print("Time taken to train: " + str("%.2f" % (end - start)) + " seconds")
        plt.plot(accuracy, label="Train Accuracy")
        plt.plot(ys, label="Test Accuracy")
        plt.legend()
        plt.title("Cost Over time")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy")
        plt.show()

        continueInput = input("Continue Training? (y/n): ")
        if continueInput == "n":
            continueInput = input("Save Weights to \"SavedWeights.txt\" (y/n)")
            if continueInput == "y":
                save_weights(input_neurons, hidden_neurons, hidden_bias, output_bias)
                break
            else:
                break
        return


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

# TODO: ADD SAVE FUNCTION TO PREDICTION VALUES AND RUN BEST VARIABLES
nEpochs = 30
batchSize = 2
learningRate = 10

for i in range(1, len(sys.argv)):
    print(sys.argv[i])

# Load training data
print("\nLoading Training Set... ")
trainSet = np.loadtxt(trainSet, float, delimiter=",")
print("Training Set Loaded. ")

print("\nLoading Training Labels... ")
trainLabel = np.loadtxt(trainLabel, int, delimiter=",")
print("Training Labels Loaded. ")

print("\nLoading Test Set... ")
try:
    testSet = np.loadtxt(testSet, float, delimiter=",")
except:
    pass
print("Test Set Loaded. ")

print("\nLoading Test Labels... ")
try:
    testLabel = np.loadtxt(testLabel, int, delimiter=",")
except:
    pass
print("Test Labels Loaded. ")

if batchSize > len(trainSet):
    print("\nERROR: Batch Size larger than Number of inputs")
    exit(1)

train_neural_net()
