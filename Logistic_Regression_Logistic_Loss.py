import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)


pos = y_train == 1
neg = y_train == 0
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="malignant")
ax.scatter(x_train[neg], y_train[neg], marker='o', s=100, c = 'b')
ax.set_ylim(-0.075,1.1)
ax.set_ylim(-0.075,1.1)
ax.set_ylabel('y')
ax.set_xlabel('Tumor Size')
plt.show()


plt.close('all')



wx, by = np.meshgrid(np.linspace(-6,12,50), np.linspace(10, -20, 40))


plt_logistic_squared_error(x_train,y_train)
plt.show()