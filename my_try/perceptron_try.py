# author:BenNoha
# datetime:2021/2/8 11:13
# software: PyCharm

"""
File description：
test perceptron
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from my_try.normal_perceptron import normal_perceptron

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# deal with label column
data = np.array(df.iloc[:100, [0, 1, -1]])
X , y = data[:,:-1] , data[:,-1]
ll = list()
for i in y:
	if i==1:
		ll.append(1)
	else :
		ll.append(-1)
y = np.array(ll)

#train
lr = 0.01
params = normal_perceptron().train(X_train=X , y_train=y,learning_rate=lr)
print(params)

x_point = np.linspace(4,7,10)
# wx+b = 0 -> w1x1+w2x2+b = 0 ->x2=-(w1x+b)/w2
w_hat = params['w']
w1_hat = w_hat[0]
w2_hat = w_hat[1]
b_hat = params['b']
y_point_hat = -(w1_hat * x_point + b_hat)/w2_hat
# draw divided line of 2 classes
plt.plot(x_point,y_point_hat)

# draw original points
aa = df[:50]['sepal length']
bb = df[:50]['sepal width']
plt.scatter(aa, bb, c='red', label='0')
cc = df[50:100]['sepal length']
dd = df[50:100]['sepal width']
plt.scatter(cc, dd, c='green', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

plt.show()
