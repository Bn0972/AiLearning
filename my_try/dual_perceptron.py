# author:BenNoha
# datetime:2021/2/8 12:09
# software: PyCharm

"""
File description：
dual perceptron
more efficency
"""

import numpy as np

# gram matrix
# update from wi,b to ai

# cost = sum(yi*(w*xi+b)) , i=1 to M , M missclassification case
# lr -> learning rate
# wi = wi + lr * yi * xi
# b = b + lr * yi

class dual_perceptron:
	def __init__(self):
		pass

	def sign(self, x, w, b):
		return np.dot(x, w) + b

	def train(self, X_train, y_train, learning_rate):
		w, b = self.initialize_with_zeros(X_train.shape[1])

		exis_wrong = True
		while exis_wrong:
			wrong_count = 0
			for i in range(len(X_train)):
				xi = X_train[i]
				yi = y_train[i]

				if yi * self.sign(xi, w, b) <= 0:
					w = w + learning_rate * np.dot(yi, xi)
					b = b + learning_rate * yi
					wrong_count += 1
			if wrong_count == 0:
				exis_wrong = False
				print('Now , there is no missclassification anymore !')

		params = {'w': w, 'b': b}
		return params

	def initialize_with_zeros(self, dim):
		w = np.zeros(dim)
		b = 0.0
		return w, b