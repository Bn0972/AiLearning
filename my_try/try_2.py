# author:BenNoha
# datetime:2021/2/17 15:25
# software: PyCharm

import numpy as np  # 导入numpy库
import time  # 导入时间库

# a = np.array([1, 2, 3, 4])  # 创建一个数据a
# print(a)
# [1 2 3 4]


# a = np.random.rand(1000000)
# b = np.random.rand(1000000)  # 通过round随机得到两个一百万维度的数组
# tic = time.time()  # 现在测量一下当前时间
# # 向量化的版本
# c = np.dot(a, b)
# toc = time.time()
# # 打印一下向量化的版本的时间
# print(c)
# print('Vectorized version:' + str(1000 * (toc - tic)) + 'ms')
#
# # 继续增加非向量化的版本
# c = 0
# tic = time.time()
# for i in range(1000000):
# 	c += a[i] * b[i]
# toc = time.time()
# print(c)
# # 打印for循环的版本的时间
# print('For loop:' + str(1000 * (toc - tic)) + 'ms')

a = np.array([[1,2,3],[4,5,6]])
b = a<3
print(f'a={a}')
print(f'b={b}')
c = np.multiply(a,b)
print(f'c={c}')
