# author:BenNoha
# datetime:2021/2/20 16:42
# software: PyCharm

"""
File description：

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# def func(x, y):
# 	return x + y
#
#
# x = np.linspace(-1, 1, 10).reshape(2,5)
# y = np.linspace(-1, 1, 10).reshape(2,5)
#
# xv, yv = np.meshgrid(x, y)
# # print(xv)
# # print(yv)
# z = func(xv, yv)
# colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
# cmap = ListedColormap(colors[:len(np.unique(z))])
# plt.contourf(xv, yv, z, cmap=cmap)
#
# # plt.scatter(x=xv, y=yv, c='b')
# plt.show()

#导入模块

#建立步长为0.01，即每隔0.01取一个点
step = 0.01
x = np.arange(-10,10,step)
y = np.arange(-10,10,step)
#也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份

#将原始数据变成网格数据形式
X,Y = np.meshgrid(x,y)
#写入函数，z是大写
Z = X**2+Y**2
#设置打开画布大小,长10，宽6
plt.figure(figsize=(10,6))
#填充颜色，f即filled
plt.contourf(X,Y,Z)
#画等高线
plt.contour(X,Y,Z)
plt.show()
