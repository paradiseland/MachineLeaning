# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei CHEN
Email:cxw19@mails.tsinghua.edu.cn
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
"""
选择算法步骤：
1.选择特征并收集训练样本
2.选择度量性能的指标
3.选择分类器并优化算法
4.评估模型的性能
5.调整算法
"""

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
# print(np.unique(y))
# stratify : provide the same classification of different y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print(accuracy_score(y_test, y_pred))
print(ppn.score(X_test_std, y_test))




if __name__ == "__main__":
    pass
