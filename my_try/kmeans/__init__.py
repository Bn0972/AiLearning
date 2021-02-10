#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from numpy import *

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

from my_try.kmeans import kMeans

if __name__ == '__main__':
	data, _ = load_iris(return_X_y=True)
	X = data[:, [2, 3]]
	k = 2
	colors = ['k', 'b', 'r']
	# centroids_orig = kMeans.randCent(dataMat=X, k=2)
	# centroids_orig = np.array(centroids_orig)
	# print(centroids_orig)
	# x1 = X[0]
	# x2 = X[1]
	# dis = kMeans.distEclud(x1, x2)
	# print(f'x1={x1},x2={x2},dis={dis}')
	centroids, cluster_assign, iter_times = kMeans.kMeans(dataMat=X, k=k)
	centroids = np.array(centroids, dtype=int)
	cluster_assign = np.array(cluster_assign)
	# print(f'cluster_assign={cluster_assign}')
	print(f'centroid={centroids}')
	print(f'iter_times = {iter_times}')
	# centers = [[4.92525253 1.68181818],[1.49215686 0.2627451 ]]

	# plt.scatter(X[:, 0], X[:, 1], c='k')

	for i in range(k):
		cluster_data = X[cluster_assign[:, 0] == i]
		plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i])
		plt.scatter(centroids[i:, 0], centroids[i:, 1], c='y', marker='v')
	plt.show()

# dataMat = mat(kMeans.loadDataSet('../../../../data/k-means/testSet.txt'))
# print('min(dataMat[:, 0])', min(dataMat[:, 0]), '\n')
# print('min(dataMat[:, 1])', min(dataMat[:, 1]), '\n')
# print('max(dataMat[:, 0])', max(dataMat[:, 0]), '\n')
# print('max(dataMat[:, 1])', max(dataMat[:, 1]), '\n')
# print(kMeans.randCent(dataMat, 2),'\n')
# print(kMeans.distEclud(dataMat[0],dataMat[1]))
# centroids, clusterAssment = kMeans.kMeans(dataMat, 4)
# print('centroids:\n', centroids, '\n')
# print('clusterAssment:\n',clusterAssment, '\n')
# dataMat3 = mat(kMeans.loadDataSet('../../../../data/k-means/testSet2.txt'))
# centList, myNewAssments = kMeans.biKmeans(dataMat3, 3)
# print('centList: \n', centList, '\n')

# fileName = '../../../../data/k-means/places.txt'
# imgName = '../../../../data/k-means/Portland.png'
# kMeans.clusterClubs(fileName=fileName, imgName=imgName, numClust=5)
