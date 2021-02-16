# author:BenNoha
# datetime:2021/2/12 13:53
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

from my_try.knn.KNearestNeighbor import KNearestNeighbor

if __name__ == '__main__':
	knn_classifier = KNearestNeighbor()
	X_train, y_train, X_test, y_test = knn_classifier.create_train_test()
	best_k = knn_classifier.cross_validation(X_train, y_train)
	dists = knn_classifier.compute_distances(X_test)
	y_test_pred = knn_classifier.predict_labels(dists, k=best_k)
	y_test_pred = y_test_pred.reshape((-1, 1))
	num_correct = np.sum(y_test_pred == y_test)
	accuracy = float(num_correct) / X_test.shape[0]
	print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))

	# aa = np.array([[27.3, 24], [24, 23.2]])
	# bb = np.linalg.inv(aa)
	# mu1 = np.array([[3,3.6]])
	# mu2 = np.array([[3.3, 2]])
	# mu = mu1-mu2
	# print(mu@bb)
