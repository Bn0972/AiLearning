# author:BenNoha
# datetime:2021/2/8 20:51
# software: PyCharm

"""
File description：

"""

import numpy as np
import icecream as ic



# data_dict = {-1: np.array([[1, 7],
# 						   [2, 8],
# 						   [3, 8]])
# 	,
# 			 1: np.array([[5, 1],
# 						  [6, -1],
# 						  [7, 3]])
# 			 }
# all_data = []
# for yi in data_dict:
# 	for fset in data_dict[yi]:
# 		for f in fset:
# 			all_data.append(f)
#
# v = np.array([3,4])
# result = np.linalg.norm(v)
# print(result)
# print(all_data)

def distance(x, z):
	# print((x-z))
	# print((x-z).T)
	return np.sqrt((x - z) @ np.transpose(x - z))


c = np.array([0, 2])
a = np.array([[11, 2, 3, 4], [12, 3, 4, 5], [13, 2, 3, 4], [14, 3, 4, 5]])
print(np.mean(a[c], axis=0))
print(a)
# xx = np.array([[1, 2, 3, 4]])
# yy = np.array([[3, 4, 5, 6]])
# rrr = pow((xx - yy), 2)
# print(rrr)


# adddd = np.array([1,2,3,4])
# ic.contextmanager()


