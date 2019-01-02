"""
This file contains implementations of computing the partial derivatives such as dc_db and dc_dw
where c is the cost function, b are the biases, w for the weights. (View course exercise for more details)
This demonstrates the use of the multivariate chain rule to train a neural network with 1 input node and 1 output node.
The output of the neural network is the NOT function i.e when x = 0, then y = 1 and when x = 1 then y = 0.
"""

import numpy as np

# Sigma function
sigma = np.tanh


# feed-forward equation
def a1(w1, b1, a0):
    z = w1 * a0 + b1
    return sigma(z)


# Individual cost function is the square of the difference between
# the network output and the training data output
def cost_function(w1, b1, x, y):
    return (a1(w1, b1, x) - y) ** 2


# This function returns the derivative of the cost function with respect to the weight
def dc_dw(w1, b1, x, y):
    z = w1 * x + b1
    dc_da = 2 * (a1(w1, b1, x) - y)  # derivative of cost with activation
    da_dz = 1 / np.cosh(z) ** 2  # derivative of activation with weighted sum z
    dz_dw = x  # derivative of weighted sum z with weight
    return dc_da * da_dz * dz_dw  # Return the chain rule product


# This function returns the derivative of the cost function with respect to the bias
def dc_db(w1, b1, x, y):
    z = w1 * x + b1
    dc_da = 2 * (a1(w1, b1, x) - y)
    da_dz = 1 / np.cosh(z) ** 2
    dz_db = 1  # derivative of weighted sum z with bias
    return dc_da * da_dz * dz_db


# test

# Initial weight and bias
w = 2.3
b = -1.2

# Single data point
x1 = 0  # input
y1 = 1

# Outputs how the cost would change in proportion to a small change in the bias
print("Change in cost with small change in bias: ", dc_db(w, b, x1, y1))

# Outputs how the cost would change in proportion to a small change in the weight
print("Change in cost with small change in weight: ", dc_dw(w, b, x1, y1))


# Below, the general case is considered i.e more neurons are added to the network.
# So now the weights are a matrix and the biases are vectors
def a1_general(weights, biases, a0):
    z = weights @ a0 + biases
    return sigma(z)


def cost_function_general(weights, biases, x, y):
    d = a1_general(weights, biases, x) - y  # Vector difference between observed and expected activation
    return d @ d  # Absolute value squared of the difference


# test

# Initial weight and bias
weight_matrix = np.array([[-0.94529712, -0.2667356, -0.91219181],
                          [2.05529992, 1.21797092, 0.22914497]])
bias = np.array([0.61273249, 1.6422662])

# training example
x_train = np.array([0.7, 0.6, 0.2])
y_train = np.array([0.9, 0.6])

print("Cost function in the general case: ", cost_function_general(weight_matrix, bias, x_train, y_train))
