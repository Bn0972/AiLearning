# author:BenNoha
# datetime:2021/2/20 17:17
# software: PyCharm

"""
File description：
An example of decision tree classifier
Test dtc on 3 dataset(moon,circle,binary)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier

# make_classification生成随机二分型数据
X, y = make_classification(n_samples=100,
						   n_features=2,
						   n_redundant=0,
						   n_informative=2,
						   random_state=1,
						   n_clusters_per_class=1)
# print(f'X={X}')
# print(f'y={y}')
# plt.scatter(X[:,0],X[:,-1],c='b')

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
# plt.scatter(X[:,0],X[:,-1],c='r')
# plt.show()
linearly_seperable = (X, y)

# prepare 3 dataset and put them into datasets

datasets = [make_moons(noise=0.3, random_state=0),
			make_circles(noise=0.2, factor=0.5, random_state=1),
			linearly_seperable]

# A, B = make_moons(noise=0.3, random_state=0)
# C, D = make_circles(noise=0.2, factor=0.5, random_state=1)
# plt.scatter(A[B == 0, 0], A[B == 0, -1], c='r')
# plt.scatter(A[B == 1, 0], A[B == 1, -1], c='b')
# plt.show()
# plt.scatter(C[D == 0, 0], C[D == 0, -1], c='r')
# plt.scatter(C[D == 1, 0], C[D == 1, -1], c='b')
# plt.show()

# 画出3种数据集和三棵决策树的分类效应图像
figure = plt.figure(figsize=(6, 9))
# i用来安排图像显示位置
i = 1
# iteration of datasets
for ds_index, ds in enumerate(datasets):
	# 对X中的数据标准化处理，然后分训练集测试集
	X, y = ds
	X = StandardScaler().fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4
														, random_state=42)

	# 找出数据集中两个特征的最大值最小值，最大值+0.5，最小值-0.5，
	# 创造一个比两个特征值更大一点的区间
	x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
	x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

	# 使用meshgrid()创建网格数据，用来准备绘制等高线（带填充色的）contourf()
	# 其中array1，array2代表网格点的横纵坐标（第一维度，第二维度）
	array1, array2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2),
								 np.arange(x2_min, x2_max, 0.2))
	# 生成画布
	# #FF0000 - 正红 , #0000FF - 正蓝
	cm = plt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])

	# 在画布加上一个子图，数据为len(datasets)行，2列，放在位置i上
	ax = plt.subplot(len(datasets), 2, i)

	if ds_index == 0:
		ax.set_title('Input data')

	# 将数据集的分布放到坐标系中，先训练集，再测试集
	ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train
			   , cmap=cm_bright, edgecolors='k')
	ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test
			   , cmap=cm_bright, alpha=0.6, edgecolors='k')

	# 为图设置坐标轴的最大最小值，并且设定没有坐标轴
	ax.set_xlim(array1.min(), array1.max())
	ax.set_ylim(array2.min(), array2.max())
	ax.set_xticks(())
	ax.set_yticks(())

	i += 1
	# i在此处取值分别为2，4，6
	ax = plt.subplot(len(datasets), 2, i)

	clf = DecisionTreeClassifier(max_depth=5)
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	concate_result = np.c_[array1.ravel(), array2.ravel()]
	# Z = clf.predict_proba(concate_result)
	Z = clf.predict(concate_result)
	# Z = Z[:,1]

	Z = Z.reshape(array1.shape)
	ax.contourf(array1, array2, Z, cmap=cm, alpha=.8)
	# 将训练集放入图中
	ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
	# 将测试集放入图中
	ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=.6)

	ax.set_xlim(array1.min(), array1.max())
	ax.set_ylim(array2.min(), array2.max())
	ax.set_xticks(())
	ax.set_yticks(())
	if ds_index == 0:
		ax.set_title('Decision Tree')

	ax.text(array1.max() - .3, array2.min() + .3, ('{:.1f}%'.format(score * 100))
			, size=15, horizontalalignment='right')
	i += 1

plt.tight_layout()
plt.show()
