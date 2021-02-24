# author:BenNoha
# datetime:2021/2/24 12:28
# software: PyCharm

"""
File description：
嵌入法
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC

train_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\digit_data\digit_data_train.csv')
X_raw = train_data.iloc[:, 1:]
Y = train_data.iloc[:, 0]
X_raw = X_raw.values
Y = Y.values.reshape(-1, 1)
print(X_raw.shape)
print(Y.shape)

threshold = 0.00067
rfc = RFC(n_estimators=10, random_state=0)
selector = SelectFromModel(rfc, threshold=threshold)
X_embedded = selector.fit_transform(X_raw, Y.flatten())
print(X_embedded.shape)
print(cross_val_score(rfc, X_embedded, Y.flatten(), cv=5).mean())

# 通过学习曲线来确定threshold的合适值
# 也可以在第一条学习曲线确定一个threshold大致范围后，画出第二条学习曲线来细化最佳的threshold
# print(rfc.fit(X_raw , Y.flatten()).feature_importances_)
# max_limit = rfc.fit(X_raw, Y.flatten()).feature_importances_.max()
# threshold_list = np.linspace(0, max_limit, 20)
# score_list = []
# for i in threshold_list:
# 	X_ebd = SelectFromModel(rfc, threshold=i).fit_transform(X_raw, Y.flatten())
# 	once = cross_val_score(rfc, X_ebd, Y.flatten(), cv=5).mean()
# 	score_list.append(once)
# print(f'threshold_list = {threshold_list}')
# print(f'score_list={score_list}')
# plt.plot(threshold_list, score_list)
# plt.show()
# best threshold = 0.00067 , score > 0.93
