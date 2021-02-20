# author:BenNoha
# datetime:2021/2/20 16:42
# software: PyCharm

"""
File description：

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def func(x, y):
	return x + y


x = np.linspace(-1, 1, 10).reshape(2,5)
y = np.linspace(-1, 1, 10).reshape(2,5)

xv, yv = np.meshgrid(x, y)
# print(xv)
# print(yv)
z = func(xv, yv)
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(z))])
plt.contourf(xv, yv, z, cmap=cmap)

# plt.scatter(x=xv, y=yv, c='b')
plt.show()
