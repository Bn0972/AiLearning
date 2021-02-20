# author:BenNoha
# datetime:2021/2/19 19:18
# software: PyCharm

"""
File description：

"""
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt

# data = (178,13) , target = (178,)
wine = load_wine(return_X_y=False)
# X_train(124,13), X_test(54,13)
X_train, X_test, Y_train, Y_test = train_test_split(wine.data
													, wine.target
													, test_size=0.3
													, random_state=30
													, shuffle=True)
# criterion:'gini'for gini inpurity , 'entropy' for information gain
# splitter = 'best'/'random'
# try max_depth from 3
# min_sample_leaf from 5
# min_sample_split from 5
# max_features
# min_inpurity_decrease
# class_weight,min_weight_fraction_leaf
clf = tree.DecisionTreeClassifier(criterion='entropy'
								  , random_state=30
								  , splitter='random'
								  , max_depth=6
								  # ,min_samples_leaf=6
								  # ,min_samples_split=6
								  )
clf.fit(X_train, Y_train)
# X_train(124,13), X_test(54,13)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
feature_name_arr = wine.feature_names.copy()
class_name_arr = wine.target_names.copy()
print(f'train_score={train_score}')
print(f'test_score={test_score}')

# apply() return the index of node where test data in the decision tree
# root index is 0
result_apply_X_test = clf.apply(X_test)
# predict() return the prediction(class) of test data
result_predict_X_test = clf.predict(X_test)
print(f'result_apply_X_test={result_apply_X_test}')
print(f'result_predict_X_test={result_predict_X_test}')
unique_apply_X_test = np.unique(result_apply_X_test)
print(f'unique_apply_X_test={unique_apply_X_test}')
# result_X_test_predict_proba = clf.predict_proba(X_test)
# print(f'result_X_test_predict_proba={result_X_test_predict_proba}')



dot_data = tree.export_graphviz(clf
								, feature_names=feature_name_arr
								, class_names=class_name_arr
								, filled=True
								, rounded=True
								)
graph = graphviz.Source(dot_data)
graph.render()

# draw learning curve
# score_list = []
# for i in range(10):
# 	clf = tree.DecisionTreeClassifier(criterion='entropy'
# 									  , random_state=30
# 									  , splitter='random'
# 									  , max_depth=i + 1
# 									  # ,min_samples_leaf=6
# 									  # ,min_samples_split=6
# 									  )
# 	clf.fit(X_train, Y_train)
# 	test_score = clf.score(X_test, Y_test)
# 	score_list.append(test_score)
#
# plt.plot(range(1,11) , score_list , c = 'red',label = 'max_depth')
# plt.legend()
# plt.show()


# result = zip(feature_name_arr, clf.feature_importances_)
# print(f'clf.feature_importances_= {clf.feature_importances_}')
# print(list(result))

# graph = graphviz.Source(dot_data)


# print(feature_name_arr)
# print(class_name_arr)


# print('*' * 50)
# print(wine_pd.columns)
# wine_pd.columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'target']
# print(wine_pd.columns)
# print(data.shape)
# print(target.shape)
# print(type(data))
# print(type(target))

# for i in range(5):
# 	print(f'data[i={i}] = {data[i]} , target[i={i}] = {target[i]}')

# a = np.array([[1,1,3,4,5,7,10,7,7]])
# one = np.unique(a, return_counts=True, return_index=True, return_inverse=True)
# print(f'a={a}')
# print(f'one={one}')
