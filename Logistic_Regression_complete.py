############### student gets admitted into a university.####################
###################### LOGISTIC REGRESSION ###############################


import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from utils import *

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

######################### Plot examples ################################
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

def sigmoid(z):
    g = 1/ (1 + np.exp(-z) )
    return g

value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    cost = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        cost = cost - y[i] * np.log(f_wb_i) - ((1 - y[i]) * np.log(1 - f_wb_i))
    total_cost = cost / m
    return total_cost 

m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):
        z_wb = sigmoid(np.dot(X[i], w) + b) - y[i]
        dj_db_i = z_wb    
        dj_db += dj_db_i
        for j in range(n):
            dj_dw[j] = dj_dw[j] + z_wb * X[i,j]
            
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    ### END CODE HERE ###

        
    return dj_db, dj_dw

initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()


def predict(X, w, b): 
    m, n = X.shape   
    p = np.zeros(m)   
    for i in range(m):
        f_wb =  sigmoid(np.dot(X[i], w) + b)
        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

p = predict(X_train, w,b)
print(f'Train Accuracy: {np.mean(p == y_train) * 100}')