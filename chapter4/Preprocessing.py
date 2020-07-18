# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei CHEN
Email:cxw19@mails.tsinghua.edu.cn
"""
import pandas as pd
from io import StringIO
import numpy as np

"""识别缺失值"""
csv_data = \
 '''A, B, C, D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,,8.0
10.0, 11.0, 12.0,'''
df = pd.read_csv(StringIO(csv_data))

print(df.isnull().sum())
print(df.values)  # 可以访问DataFrame底层numpy阵列钟鼎文数据

"""删除缺失值"""
# df.dropna(axis=0)  # 删除行
# df.dropna(how='all')  # 删除所有列都是NaN的行
# df.dropna(thresh=4)  # 删除少于4个实数的行
# df.dropna(subset=['C'])  # 只删除C列出现NaN的行

"""填补缺失值"""
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)


df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
"""
特征：
    名词特征
    序数特征
    数值特征
"""
"""映射序数特征"""
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)

"""分类标签编码"""
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

"""名词特征做热编码"""
# 一列变成三列特征， 产生多重共线性，所以要删除一列
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
ohe = OneHotEncoder()
ohe.fit_transform(X[:, 0].reshpe(-1, 1)).toarray()

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)

"""把特征保持在统一尺度熵"""

"""
归一化:
    x = (x(i)-x_min)/(x_max-x_min)
标准化:
    x = (x(i)-mu)/sigma
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X_train = []
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)

stdsc = StandardScaler()
X_train_srd = stdsc.fit_transform(X_train)

# 若模型在训练集熵表现远比测试集好，，很可能是发生了过拟合
"""
解决方案：
收集更多训练数据
通过正则化引入对复杂性的惩罚
选择参数较少的简单模型
减少数据的维数
"""

"""
L1 : ||w||_1 = Sigma: |w_j|  会产生很多特征权重为0的特征向量， =>特征选择技术
L2 : ||w||_2 = Sigma: w_j**2 惩罚权重大的个体来降低模型复杂度

加强正则化可以减少稀疏性， 选择较低的C值
"""


"""
特征选择
贪婪选择；
逆顺序选择；
随机森林 　forest.feature_importances_
"""
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=.1, prefit=True)




if __name__ == "__main__":
    pass
