# author:BenNoha
# datetime:2021/2/22 13:36
# software: PyCharm

"""
File description：

"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler

# Scalers
scaler = MinMaxScaler(feature_range=(0, 1))
raw_X = np.array([[-1, 2], [-.5, 6], [0, 10], [1, 18]])
# print(f'raw_X={raw_X}')
# scaled_X = scaler.fit_transform(X=raw_X)
# print(f'scaled_X = {scaled_X}')
# print(scaler.data_max_)
# print(scaler.data_min_)
# inverse_X = scaler.inverse_transform(scaled_X)
# print(f'inverse_X = {inverse_X}')
# # scaler.partial_fit()
# result = scaler.transform([[2,2]])
# print(f'result={result}')
# st_scaler = StandardScaler()
# std_X = st_scaler.fit_transform(X=raw_X)
# print(f'st_X = {std_X}')
# print(st_scaler.mean_)
# print(st_scaler.var_)
# print(np.mean(std_X,axis=0))
# print(np.var(std_X,axis=0))
#
# max_scaler = MaxAbsScaler()
# print(f'raw_X={raw_X}')
# print(f'MaxAbsScaler(raw_X)={max_scaler.fit_transform(raw_X)}')
# robust_scaler = RobustScaler()

# Imputer
# from sklearn.impute import SimpleImputer
# data1 = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
# sim_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# sim_imputer.fit(data1)

# LabelEncoder
# from sklearn.preprocessing import LabelEncoder , OrdinalEncoder , OneHotEncoder
# enc = OneHotEncoder()
# XX = [['Male', 1], ['Female', 3], ['Female', 2]]
# enc.fit(XX)
# print(enc.categories_)
# XX_result = enc.transform(XX).toarray()
# print(XX_result)
# print(enc.get_feature_names())

# Binarizer
from sklearn.preprocessing import Binarizer , KBinsDiscretizer
# bnr = Binarizer(threshold=1)

aa = set('google')
print(aa)