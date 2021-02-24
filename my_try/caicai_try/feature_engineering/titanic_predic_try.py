# author:BenNoha
# datetime:2021/2/23 14:55
# software: PyCharm

"""
File description：

"""
import pandas as pd

train_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\titanic\titanic_train.csv',index_col=['PassengerId'])
test_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\titanic\titanic_test.csv',index_col=['PassengerId'])
print(type(train_data))
print(type(test_data))
# print(train_data.info)
# print('*'*50)
# print(test_data.info())