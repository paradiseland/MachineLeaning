# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei CHEN
Email:cxw19@mails.tsinghua.edu.cn
"""

from sklearn.neighbors import KNeighborsClassifier
from Perceptron import X_train_std, y_train

# p=1 曼哈顿距离， p=2 欧几里得距离
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

"""
步骤：
    1.选择k个数和一个距离度量
    2.找到分类样本的k-近邻
    3.以多数票机制确定分类标签
    
新数据点的分类标签由最靠近该点的k个数据点的多数票决定
"""



if __name__ == "__main__":
    pass
