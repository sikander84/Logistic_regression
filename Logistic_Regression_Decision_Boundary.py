import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

z = np.arange(-10,11)

tmp_z = sigmoid(z)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, tmp_z, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
plt.show()


x0 = np.arange(0, 6)
x1 = 3 - x0

fig,ax = plt.subplots(1, 1, figsize=(5, 4) )
ax.plot(x0,x1, c = "b" )
ax.axis([0, 4, 0, 3.5])

ax.fill_between(x0,x1, alpha=0.2)
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

