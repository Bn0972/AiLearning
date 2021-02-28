# datetime:2021/2/28 13:41
# software: PyCharm

"""
File description：
回归树
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 506*13
boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
# score默认为R2-score ， 或者neg_mean_squared_error
score_result = cross_val_score(estimator=regressor,X=boston.data , y=boston.target,
				cv=10,scoring='neg_mean_squared_error')
# print(score_result)
# print(boston.DESCR)
# print(boston.feature_names)
# print(boston.filename)

