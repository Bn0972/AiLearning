# author:BenNoha
# datetime:2021/2/8 20:35
# software: PyCharm

"""
File description：
hard margin svm
"""

import numpy as np
import matplotlib.pyplot as plt


class hard_margin_svm:
	def __init__(self, visualization=True):
		self.visualization = visualization
		colors = {1: 'r', -1: 'g'}
		if self.visualization:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)

	def train(self, data):
		# data_dict looks like: {abs(w) , [w,b]}
		opt_dict = {}
		transforms = [[1, 1],
					  [-1, 1],
					  [-1, -1],
					  [1, -1]]

	def predict(self, p):
		pass

	def visualize(self):
		pass

	def hyperplane(self, x, w, b, v):
		pass
