# author:BenNoha
# datetime:2021/2/16 20:38
# software: PyCharm

"""
File description：
Test numpy.einsum()
References:http://www.atyun.com/32288.html
"""
import numpy as np

a = np.arange(9).reshape((3, 3))
b = np.arange(start=1, stop=10).reshape((3, 3))


def spliter(str):
	print('*' * 50)
	print(str)
	print(f'a=\n{a}')
	print(f'b=\n{b}')


# just output A
spliter("just output A")
print(f"np.einsum('ij',a) =\n {np.einsum('ij', a)}")
# A transpose
spliter("A transpose")
print(f"np.einsum('ji',a) =\n {np.einsum('ji', a)}")
# A diag
spliter("A diag")
print(f"np.einsum('ii',a) =\n {np.einsum('ii->i', a)}")

# trace of A
spliter("trace of A")
print(f"np.einsum('ii->',a) =\n {np.einsum('ii->', a)}")

# sum all item of A
spliter("sum of A")
print(f"np.einsum('ij->',a) =\n {np.einsum('ij->', a)}")

# sum of A by column
spliter("sum of A by column")
print(f"np.einsum('ij->j',a) =\n {np.einsum('ij->j', a)}")

# sum of A by row
spliter("sum of A by row")
print(f"np.einsum('ij->i',a) =\n {np.einsum('ij->i', a)}")

# multiply every correspondent item of A and B
spliter("multiply every correspondent item of A and B")
print(f"np.einsum('ij,ij->ij',a,b) =\n {np.einsum('ij,ij->ij', a, b)}")

# multiply every correspondent item of A and B.T
spliter("multiply every correspondent item of A and B.T")
print(f"np.einsum('ij,ji->ij',a,b) =\n {np.einsum('ij,ji->ij', a, b)}")

# dot(A,B)
spliter("dot(A,B)")
print(f"np.einsum('ij,jk->ik',a,b) =\n {np.einsum('ij,jk->ik', a, b)}")

# inner(A,B) A row multipy with B row
spliter("inner(A,B) A row multipy with B row")
print(f"np.einsum('ij,kj->ik',a,b) =\n {np.einsum('ij,kj->ik', a, b)}")

# every row in A multipy with B
spliter("every row in A multipy with B ")
print(f"np.einsum('ij,kj->ikj',a,b) =\n {np.einsum('ij,kj->ikj', a, b)}")

# every item in A multipy with B
spliter("every item in A multipy with B ")
result = np.einsum('ij,kl->ijkl', a, b)
print(f'shape = {result.shape}')
print(f"np.einsum('ij,kl->ijkl',a,b) =\n {result}")

# print(np.einsum('ij,jk->ijk', a, b))
