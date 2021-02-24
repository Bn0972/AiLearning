# author:BenNoha
# datetime:2021/2/23 15:26
# software: PyCharm

"""
File description：
Example using digit_dataset to filter features
过滤法
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, \
	chi2, f_classif, f_regression, mutual_info_classif as MIC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 28*28 pixels for each image
# 28*28 = 784 , train_data.shape=(42000,1+784)
# digit number(1 column) + pixels(784) = 785 columns
train_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\digit_data\digit_data_train.csv')
# test_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\digit_data\digit_data_test.csv')
# pd.set_option('display.max_columns',None)
print(train_data.columns)

X_raw = train_data.iloc[:, 1:]
Y = train_data.iloc[:, 0]
X_raw = X_raw.values
Y = Y.values.reshape(-1, 1)
print(X_raw.shape)
print(Y.shape)

selector = VarianceThreshold(threshold=0)
X_var = selector.fit_transform(X=X_raw)
# (42000,784) -> (42000,708)
print(X_var.shape)

var_of_X_raw = np.var(X_raw, axis=0)
# print(var_of_X_var.shape)
median_of_var_of_X_raw = np.median(var_of_X_raw)
print(median_of_var_of_X_raw)
selector2 = VarianceThreshold(threshold=median_of_var_of_X_raw)
# (42000,784) -> (42000,392 )
X_var_2 = selector2.fit_transform(X_raw)
print(X_var_2.shape)

# 伯努利分布 var[X] = p(1-p), p=0.8
# X_bvar = VarianceThreshold(0.8*(1-0.8)).fix_transform(X)
# pca_trainer = PCA()

# 卡方检验
# After filtering by variance , we can use Chi test filer
# Make sure every value in X_var_2 is non_negative
# Before chi2
# scores1 = cross_val_score(RFC(n_estimators=10, random_state=0), X_var_2, Y.flatten(), cv=5)
# print(scores1)
# print(scores1.mean())

# X_chi2 = SelectKBest(chi2, k=300).fit_transform(X_var_2, Y)
# print(X_chi2.shape)


# After chi2
# scores2 = cross_val_score(RFC(n_estimators=10, random_state=0), X_chi2, Y.flatten(), cv=5)
# print(scores2)
# print(scores2.mean())

# Get and show the chi2() result which contains chi-value and p-value

# chiValue, pValue = chi2(X=X_var_2, y=Y.flatten())

# What we want is best k : features no.(now) - column no. of(p-value>0.05)
# pvalue < 0.05(或者0.01) , 拒绝原假设（特征X与标签Y独立） , 接受备用假设(X与Y相关)

# independentColumnsNo_of_chi = (pValue > 0.05).sum()
# k_best_chi = len(chiValue) - independentColumnsNo_of_chi
# print(f'No. of ignored columns = {independentColumnsNo_of_chi}')
# print(f'Best k_best = {k_best_chi}')

# To determine best k in chi2() , we draw the learning curve
# score_list = []
# k_range = range(390, 150, -10)
# for i in k_range:
# 	X_chi2_plot = SelectKBest(chi2, k=i).fit_transform(X_var_2, Y)
# 	once = cross_val_score(RFC(n_estimators=10, random_state=0), X_chi2_plot, Y.flatten(), cv=5).mean()
# 	score_list.append(once)
# plt.plot(k_range, score_list)
# plt.show()


# F检验
# F , p_value = f_classif(X_var_2,Y.flatten())
# independentColumnsNo_of_F_classif = (p_value>0.05).sum()
# k_best_F = len(F) - independentColumnsNo_of_F_classif
# print(f'independentColumnsNo_of_F_classif = {independentColumnsNo_of_F_classif}')
# print(f'k_best_F = {k_best_F}')

# 互信息法
result = MIC(X_var_2, Y.flatten())
colno = sum(result <= 0)
k_best_mutual = len(result) - colno
print(f'colno={colno}')
print(f'k_best_mutual={k_best_mutual}')
