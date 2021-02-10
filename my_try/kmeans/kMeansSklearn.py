# -*- coding:UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# 加载数据集


dataMat = []
# fr = open("data/10.KMeans/testSet.txt") # 注意，这个是相对路径，请保证是在 MachineLearning 这个目录下执行。
# for line in fr.readlines():
#     curLine = line.strip().split('\t')
#     fltLine = list(map(float,curLine))    # 映射所有的元素为 float（浮点数）类型
#     dataMat.append(fltLine)

# 训练模型
data, _ = load_iris(return_X_y=True)
X = data[:, [2, 3]]
k = 2
km = KMeans(n_clusters=k)  # 初始化
km.fit(X)  # 拟合
km_pred = km.predict(X)  # 预测
centers = km.cluster_centers_  # 质心
print(f'km_pred={km_pred} , centers = {centers} , n_iter_ = {km.n_iter_}')

# 可视化结果
plt.scatter(np.array(X)[:, 1], np.array(X)[:, 0], c=km_pred)
plt.scatter(centers[:, 1], centers[:, 0], c="r")
plt.show()