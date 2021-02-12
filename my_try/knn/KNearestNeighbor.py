# author:BenNoha
# datetime:2021/2/12 15:05
# software: PyCharm

"""
File description：

"""
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class KNearestNeighbor:
	def __init__(self):
		pass

	def train(self, X, y):
		self.X_train = X
		self.y_train = y

	def compute_distances(self, X):
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))
		M = np.dot(X, self.X_train.T)
		te = np.square(X).sum(axis=1)
		tr = np.square(self.X_train).sum(axis=1)
		dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
		return dists

	def predict_labels(self, dists, k=1):
		num_test = dists.shape[0]

		y_pred = np.zeros(num_test)
		for i in range(num_test):
			closest_y = []
			labels = self.y_train[np.argsort(dists[i, :])].flatten()
			closest_y = labels[0:k]
			c = Counter(closest_y)
			y_pred[i] = c.most_common(1)[0][0]
		return y_pred

	def cross_validation(self, X_train, y_train):
		num_folds = 5

		k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
		X_train_folds = []
		y_train_folds = []
		X_train_folds = np.array_split(X_train, num_folds)
		y_train_folds = np.array_split(y_train, num_folds)
		k_to_accuracies = {}
		for k in k_choices:
			for fold in range(num_folds):
				validation_X_test = X_train_folds[fold]
				validation_y_test = y_train_folds[fold]
				temp_X_train = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
				temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])
				self.train(temp_X_train, temp_y_train)
				temp_dists = self.compute_distances(validation_X_test)
				temp_y_test_pred = self.predict_labels(temp_dists, k=k)
				temp_y_test_pred = temp_y_test_pred.reshape((-1, 1))  # C
				num_correct = np.sum(temp_y_test_pred == validation_y_test)
				num_test = validation_X_test.shape[0]
				accuracy = float(num_correct) / num_test
				k_to_accuracies[k] = k_to_accuracies.get(k, []) + [accuracy]  # P
		for k in sorted(k_to_accuracies):
			for accuracy in k_to_accuracies[k]:
				print('k = %d, accuracy = %f' % (k, accuracy))
		accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
		best_k = k_choices[np.argmax(accuracies_mean)]
		print('最佳k值为{}'.format(best_k))
		return best_k

	def create_train_test(self):
		data_X, data_y = datasets.load_iris(return_X_y=True)
		X, y = shuffle(data_X, data_y, random_state=13)
		X = X.astype(np.float32)
		# divide training and test set
		offset = int(0.7 * X.shape[0])
		X_train, y_train = X[:offset], y[:offset]
		X_test, y_test = X[offset:], y[offset:]
		# turn an array to a matrix
		y_train = y_train.reshape((-1, 1))
		y_test = y_test.reshape((-1, 1))

		# print(f'X_train.shape={X_train.shape}')
		# print(f'X_test.shape={X_test.shape}')
		# print(f'y_train.shape={y_train.shape}')
		# print(f'y_test.shape={y_test.shape}')
		return X_train, y_train, X_test, y_test
