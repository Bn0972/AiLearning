# author:BenNoha
# datetime:2021/2/6 17:38
# software: PyCharm

"""
File description：

"""
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


# # Try svd

# A = np.array([[0, 1], [1, 1], [1, 0]])
# u, s, vh = np.linalg.svd(A, full_matrices=True)
# print(f'u={u} , s = {s} , vh = {vh}')
# print(f'u.shape={u.shape} , s.shape = {s.shape} , vh.shape = {vh.shape}')
# sa = np.diag(s)
# print(f'sa.shape = {sa.shape} , sa = {sa}')
# result = u[:,0:2]@sa@vh
# print(f'result = {result}')


# test stack with 3_dimensional array

# t = np.array([[[1, 2, -1], [3, 4, -3]],
# 			  [[5, 6, -5], [7, 8, -7]],
# 			  [[9, 10, -9], [10, 11, -10]]])
# print(t.shape)
# print(t)
# print('*'*50)
# a = t[:, :, 0]
# b = t[:, :, 1]
# c = t[:, :, 2]
# print(a)
# print('*'*50)
# print(b)
# print('*'*50)
# print(c)
# print('#'*50)
# o = np.stack((a,b,c) , axis=2)
# print(0)
# print('#'*50)
# print(t)


def restore(u, s, v, kk):
	# k singular value number

	m, n = len(u), len(v[0])
	a = np.zeros((m, n))
	for k in range(kk):
		uk = u[:, k].reshape(m, 1)
		vk = v[k].reshape(1, n)
		a += s[k] * np.dot(uk, vk)
		a = a.clip(0,255)
	return np.rint(a).astype('uint8')


# SVD with image
A = np.array(Image.open('./1.jpg', 'r'))

u_r, s_r, v_r = np.linalg.svd(A[:, :, 0])
u_g, s_g, v_g = np.linalg.svd(A[:, :, 1])
u_b, s_b, v_b = np.linalg.svd(A[:, :, 2])

K = 50

output_path = r'./svd_pic'

for k in tqdm(range(1,K+1)):
	R = restore(u_r, s_r, v_r, k)
	G = restore(u_g, s_g, v_g, k)
	B = restore(u_b, s_b, v_b, k)
	O = np.stack((R, G, B), axis=2)
	Image.fromarray(O).save('%s\\svd_img_%d.jpg' % (output_path, k))

print('end')
