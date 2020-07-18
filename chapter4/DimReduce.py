# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei CHEN
Email:cxw19@mails.tsinghua.edu.cn
"""
"""
降维压缩：
    PCA 主成分分析
    LDA 线性判别分析
    KPCA 核主成分分析
"""
"""
PCA：寻找高维数据中存在方差最大的方向
步骤：
    1.标准化d维数据集
    2.构建协方差矩阵
    3.将协方差矩阵分解为特征向量和特征值
    4.通过降阶对特征值排序，对相应的特征向量排序
    5.选择对应k个最大特征值的k个特征向量，k为新子空间维数
    6.从最上面的k特征向量开始构造投影矩阵W
    7.用投影矩阵W变化d维输入数据集X，获得新的k维特征子空间
"""
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
df_wine = sklearn.datasets.load_wine()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)










if __name__ == "__main__":
    pass
