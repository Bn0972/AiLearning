# author:BenNoha
# datetime:2021/2/24 13:35
# software: PyCharm

"""
File description：
包装法
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\digit_data\digit_data_train.csv')
X_raw = train_data.iloc[:, 1:]
Y = train_data.iloc[:, 0]
X_raw = X_raw.values
Y = Y.values.reshape(-1, 1)
print(X_raw.shape)
print(Y.shape)
print('*'*50)

rfc = RFC(n_estimators=10, random_state=0)
feature_to_select = 340
step = 50
selector = RFE(rfc , n_features_to_select=feature_to_select,step=50).fit(X_raw,Y.flatten())

# support_:bool array表示特征是否被选中
# print(selector.support_)
# print('*'*50)
# print(selector.support_.sum())
# print('*'*50)
# ranking_:表示综合排名
# print(selector.ranking_)
# print('*'*50)

# X_wrapper = selector.transform(X_raw)
# score = cross_val_score(rfc,X_wrapper,Y.flatten(),cv=5).mean()
# print(score)

feature_select_no_list=range(1,751,50)
score_list = []
for i in feature_select_no_list:
	X_wrp = RFE(rfc,n_features_to_select=i,step=50).fit_transform(X_raw,Y.flatten())
	once = cross_val_score(rfc,X_wrp,Y.flatten(),cv = 5).mean()
	score_list.append(once)
plt.figure(figsize=[20,5])
plt.plot(feature_select_no_list,score_list)
plt.xticks(feature_select_no_list)
plt.show()