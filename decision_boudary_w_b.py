import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import  plot_data, sigmoid, dlc

x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])   

pos = y_train == 0
neg = y_train == 1

x0 = np.arange(0, 6)
x1 = 3 - x0
x1_other = 4 - x0


fig,ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(x0, x1, c="r", label ="$b$=-3" )
ax.plot(x0, x1_other, c="b", label ="$b$=-4" )

plt.show()
