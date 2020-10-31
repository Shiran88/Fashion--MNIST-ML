import numpy as np
import random


# initialize the network (the weights and the bias)
def initialize_network(w, b):
    img_matrix_size = 28
    num_of_output_classes = 10
    num_of_hidden_neurons = 200
    num_of_input_neurons = img_matrix_size * img_matrix_size
    w['inputToHiddenWeight'] = np.random.uniform(low=-0.3, high=0.3,
                                                 size=(num_of_input_neurons, num_of_hidden_neurons))
    w['hiddenToOutputWeight'] = np.random.uniform(low=-0.3, high=0.3,
                                                  size=(num_of_hidden_neurons, num_of_output_classes))
    b['inputToHiddenBias'] = np.random.uniform(low=-0.3, high=0.3, size=(1, num_of_hidden_neurons))
    b['hiddenToOutputBias'] = np.random.uniform(low=-0.3, high=0.3, size=(1, num_of_output_classes))
    return w, b


# relu is an nonlinear activation function
def relu(value):
    temp_copy = np.copy(value)
    return np.maximum(temp_copy, 0, temp_copy)


def un_linear_func_prime(val):
    temp = np.copy(val)
    temp[temp <= 0] = 0
    temp[temp > 0] = 1
    return temp


# compute the gradients w.r.t all the parameters
def back_propagation(w, b, z1, h1, z2, h2, x, y):
    db2 = np.copy(h2)
    db2[0][y] -= 1
    dw2 = np.dot(h1.T, db2)
    temp = np.dot(db2, w['hiddenToOutputWeight'].T)
    temp *= un_linear_func_prime(z1)
    dw1 = np.outer(x, temp)
    db1 = temp
    return db1, dw1, db2, dw2


def update_weights_and_bias(w, b, b1, w1, b2, w2, learning_rate):
    w['inputToHiddenWeight'] -= learning_rate * w1
    w['hiddenToOutputWeight'] -= learning_rate * w2
    b['inputToHiddenBias'] -= learning_rate * b1
    b['hiddenToOutputBias'] -= learning_rate * b2
    return w, b


# normalize the vector into a probability distribution -  values will be in range 0 to 1,
# and the sum of all the probabilities will be equal to one
def soft_max(temp):
    temp -= temp.max()
    return np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)


"""
front_propagation - the input data is fed in the forward direction through the network.
Each hidden layer accepts the input data, processes it as per the
activation function and passes to the successive layer.
"""


def front_propagation(w, b, sample_x):
    z1 = np.dot(sample_x, w['inputToHiddenWeight']) + b['inputToHiddenBias']
    h1 = relu(z1)
    z2 = np.dot(h1, w['hiddenToOutputWeight']) + b['hiddenToOutputBias']
    h2 = soft_max(z2)
    return z1, h1, z2, h2


# check the loss and the accuracy rate on the validation set
def validation_loss_and_accuracy(w, b, validation_set):
    sum_loss, num_of_correct_predictions = 0.0, 0.0
    for sample in validation_set:
        # y (sample[1]) is the probability that x(sample[0]) is classified as tag y
        z1, h1, z2, h2 = front_propagation(w, b, sample[0])
        # compute the loss with Negative Log Likelihood loss function
        loss = -(np.log(h2[0][int(sample[1])]))
        sum_loss += loss
        if h2.argmax() == sample[1]:
            num_of_correct_predictions += 1
    # number of correct predictions / number of examples
    accuracy = num_of_correct_predictions / len(validation_set)
    # calculate the average loss
    avg_loss = sum_loss / len(validation_set)
    return avg_loss, accuracy


# train the network
def train_network(train_examples, w, b, learning_rate):
    # shuffle the data
    random.shuffle(train_examples)
    training_set_size = 44000
    x, y = zip(*train_examples)
    training_set = list(zip(x[:training_set_size], y[:training_set_size]))
    # set aside a small part from the data for the validation set
    validation_set = list(zip(x[training_set_size:], y[training_set_size:]))

    for epochs in range(0, 21):
        for sample in training_set:
            z1, h1, z2, h2 = front_propagation(w, b, sample[0])
            b1, w1, b2, w2 = back_propagation(w, b, z1, h1, z2, h2, sample[0], np.int(sample[1]))
            w, b = update_weights_and_bias(w, b, b1, w1, b2, w2, learning_rate)
        if epochs == 7:
            learning_rate = 0.005
        elif epochs == 12:
            learning_rate = 0.003
        np.random.shuffle(training_set)
    # check accuracy and loss on validation set
    # validation_loss, validation_accuracy = validation_loss_and_accuracy(w, b, validation_set)


# write the predictions on the test set in a file
def write_predictions_in_file(test_x, w, b):
    with open('test_y', 'w') as the_file:
        for sample_x in test_x:
            z1, h1, z2, h2 = front_propagation(w, b, sample_x)
            the_file.writelines(str(np.argmax(h2)) + '\n')


def main():
    # load train_x examples and normalize their values to the range [0,1]
    train_x = np.loadtxt("train_x") / 255.0
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    # combine train_x and train_y examples
    train_examples = list(zip(train_x, train_y))
    w = {'inputToHiddenWeight': [], 'hiddenToOutputWeight': []}
    b = {'inputToHiddenBias': [], 'hiddenToOutputBias': []}
    # initialize the network
    w, b = initialize_network(w, b)
    # set the learning rate
    learning_rate = 0.008
    # train the network (SGD)
    train_network(train_examples, w, b, learning_rate)
    # write the predictions on the test set in a file
    write_predictions_in_file(test_x, w, b)


if __name__ == "__main__":
    main()
