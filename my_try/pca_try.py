# author:BenNoha
# datetime:2021/2/7 14:40
# software: PyCharm

"""
File description：

"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class PCA:
	def compute_cov_matrix(self, X):
		m = X.shape[0]
		# Standalize
		X = X - np.mean(X, axis=0)
		return 1 / m * np.dot(X.T, X)

	def compute_pca(self, X, n_component):
		cov_m = self.compute_cov_matrix(X)
		eigenvalue, eigenvector = np.linalg.eig(cov_m)
		# sort eigentvalue from big to small
		idx = np.argsort(eigenvalue)[::-1]
		eigenvector = eigenvector[idx]
		# eigentvector.shape = n*n_component , n is size of features
		p = eigenvector[:, :n_component]
		# pca = PX (dimension of X change from m*n to m*n_component)
		return np.dot(X, p)


data = datasets.load_digits()
x = data.data
y = data.target
n_component = 2
x_trans = PCA().compute_pca(x, n_component)
x1 = x_trans[:, 0]
x2 = x_trans[:, 1]

# draw plot
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

class_distr = []

for i, l in enumerate(np.unique(y)):
	_x1 = x1[y == l]
	_x2 = x2[y == l]
	_y = y[y == l]
	class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

plt.legend(class_distr, y, loc=1)
plt.suptitle('PCA diemnsionality reduction')
plt.title('Digit dataset')
plt.xlabel('Principal component 1')
plt.xlabel('Principal component 2')
plt.show()
