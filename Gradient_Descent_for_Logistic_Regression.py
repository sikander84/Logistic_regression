import copy, math
import numpy as np
import matplotlib.pyplot as plt


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


############### Plot of graph on the input data   ############### 
"""
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()

"""

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def compute_cost_logistic(X,y,w,b):
    m = X.shape[0]
    cost = 0.
    for i  in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost = cost - (y[i] * np.log(f_wb_i)) - ((1 - y[i]) * np.log(1 - f_wb_i))
    cost = cost / m    
    return cost


def compute_gradient_logistic(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.0
    for i in range(m):
        z_1 = np.dot(X[i], w) + b 
        f_wb_i = sigmoid(z_1)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


############### Test compute_gradient_logistic   ############### 
"""
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_dw_tmp, dj_db_tmp = compute_gradient_logistic(X_train, y_train,w_tmp,b_tmp)

print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )

"""

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X,y,w,b)
        w= w - alpha * dj_dw
        b= b - alpha * dj_db
        if (i < 100000):
            J_history.append( compute_cost_logistic(X,y,w,b))
    
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history



w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")