# author:BenNoha
# datetime:2021/2/23 14:55
# software: PyCharm

"""
File description：
代码中只用到train dataset
"""
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# step 1 get data
train_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\titanic\titanic_train.csv',
						 index_col=['PassengerId'])
# test_data = pd.read_csv(r'G:\pycharm_pyproject\AiLearning\my_try\caicai_try\dataset\titanic\titanic_test.csv',
# 						index_col=['PassengerId'])

pd.set_option('display.max_columns', None)
# print(type(train_data))
# print(type(test_data))
# print(train_data.info)
# print('*'*50)
# print(train_data.info())

# Using pd dropna
# missing value number of column -> Age:177,Cabin:687,Embarked:2
# missing_value_count = train_data.isnull().sum()
# print(f'null item number in each column={missing_value_count}')


# all_entry = np.product(train_data.shape)
# missing_value_entry = missing_value_count.sum()
# print(f'ratio of missing part = {missing_value_entry/all_entry*100}%')
# train_data_dropedna = train_data.dropna(axis=1)
# print(f'train_data.shape = {train_data.shape}')
# print(f'train_data_dropedna.shape = {train_data_dropedna.shape}')

# Using pd fillna
# 设置显示最大行
# pd.set_option('display.max_columns', None)
# print(train_data.head())

# print(train_data.columns)


# step 2 preprocessing
# print(train_data.shape)
train_data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)
# print(train_data.shape)

# deal with missing value , fill and drop
# fill 'Age'
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
# drop samples(rows) which 'Embarked' is NA value
train_data = train_data.dropna()
# print(train_data.shape)
# 二分类转化为数字
train_data['Sex'] = (train_data['Sex'] == 'male').astype('int')
# 三分类转化为数字
uni_list = train_data['Embarked'].unique()
# print(type(uni_list))
# print(uni_list)
labels = uni_list.tolist()
train_data['Embarked'] = train_data['Embarked'].apply(lambda x: labels.index(x))

# 检查Sex与Embarked列
# print(train_data.head())
# print(train_data['Embarked'].unique())

# print(train_data.shape)
X_data = train_data.iloc[:, train_data.columns != 'Survived']
y_data = train_data.iloc[:, train_data.columns == 'Survived']
# print(f'X_data.shape={X_data.shape}')
# print(f'y_data.shape={y_data.shape}')
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
# print(f'X_train.shape={X_train.shape}')
# print(f'X_test.shape={X_test.shape}')
# print(f'y_train.shape={y_train.shape}')
# print(f'y_test.shape={y_test.shape}')


# ?????
for i in [X_train, X_test, y_train, y_test]:
	i.index = range(i.shape[0])

clf = DecisionTreeClassifier(random_state=25)
clf.fit(X_train, y_train)
score_ = clf.score(X_test, y_test)
print(f'score_={score_}')
score_list = cross_val_score(clf, X_data, y_data, cv=10)
print(f'score_list={score_list}')
print(f'score_list.mean()={score_list.mean()}')

# 在不同max_depth下观察拟合情况
tr = []
te = []
for i in range(10):
	clf = DecisionTreeClassifier(random_state=25
								 , max_depth=i + 1
								 , criterion='entropy'
								 )
	clf = clf.fit(X_train, y_train)
	score_tr = clf.score(X_train, y_train)
	score_te = cross_val_score(clf, X_data, y_data, cv=10).mean()
	tr.append(score_tr)
	te.append(score_te)
print(max(te))
plt.plot(range(1, 11), tr, color='red', label='train')
plt.plot(range(1, 11), te, color='blue', label='test')
plt.xticks(range(1, 11))
plt.yticks(range(0, 1))
plt.legend()
plt.show()

# 用网格搜索参数
gini_thresholds = np.linspace(0, 0.5, 20)
parameters = {'splitter': ('best', 'random')
	, 'criterion': ('gini', 'entropy')
	, 'max_depth': [*range(1, 10)]
	, 'min_samples_leaf': [*range(1, 50, 5)]
	, 'min_impurity_decrease': [*np.linspace(0, 0.5, 20)]
			  }
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf,param_grid=parameters,cv=10)
GS.fit(X_train,y_train)
print(f'GS.best_params_={GS.best_params_}')
print(f'GS.best_params_={GS.best_score_}')

print(f'GS.best_index_={GS.best_index_}')
print(f'GS.best_estimator_={GS.best_estimator_}')

'''
GS.best_params_={'criterion': 'entropy', 
				'max_depth': 3, 
				'min_impurity_decrease': 0.0,
 				'min_samples_leaf': 1,
 				 'splitter': 'best'}
GS.best_params_=0.823195084485407
GS.best_index_=4400
GS.best_estimator_=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=3, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=25, splitter='best')
'''