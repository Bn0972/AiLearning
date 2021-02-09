# author:BenNoha
# datetime:2021/2/8 20:51
# software: PyCharm

"""
File description：

"""

import numpy as np

data_dict = {-1: np.array([[1, 7],
						   [2, 8],
						   [3, 8]])
	,
			 1: np.array([[5, 1],
						  [6, -1],
						  [7, 3]])
			 }
all_data = []
for yi in data_dict:
	for fset in data_dict[yi]:
		for f in fset:
			all_data.append(f)

v = np.array([3,4])
result = np.linalg.norm(v)
print(result)
print(all_data)
