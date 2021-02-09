# author:BenNoha
# datetime:2021/2/8 20:28
# software: PyCharm

"""
File description：
try something about svm
"""

import numpy as np
import matplotlib.pyplot as plt

from my_try.hard_margin_svm import hard_margin_svm

predict_data = [[0, 10],
				[1, 3],
				[3, 4],
				[3, 5],
				[5, 5],
				[5, 6],
				[6, -5],
				[5, 8],
				[2, 5],
				[8, -3]]

data_dict = {-1: np.array([[1, 7],
						   [2, 8],
						   [3, 8]])
	,
			 1: np.array([[5, 1],
						  [6, -1],
						  [7, 3]])
			 }

svm = hard_margin_svm()
svm.train(data=data_dict)

for p in predict_data:
	svm.predict(p)
svm.visualize()




# draw data points

# for i in data_dict:
# 	for x in data_dict[i]:
# 		ax.scatter(x[0], x[1], s=100, color=colors[i])







plt.show()
