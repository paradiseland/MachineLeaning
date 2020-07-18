# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei CHEN
Email:cxw19@mails.tsinghua.edu.cn
"""

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from Perceptron import X_train_std, y_train, X_test_std, y_test, X_train
from pydotplus import graph_from_dot_data

"""
最大化每次分裂时的信息增益
IG(D_p, f) = I(D_p) - Sigma:Nj/Np*I(Dj)
        Dp: parent node
        Dj: ith kid node
        I:杂质含量
        Np:父节点样本数
        Nj:字节点j样本数
    字节点杂质含量越低，信息增益越大

三个杂质度量：
    I_G : 基尼杂质度
        I_G(t) = 1 - Sigma: p(i|t)**2, 类的分布均匀，熵值最大
    I_H : 熵
    I_E : 分类错误
    
"""

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)

dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginica'],
                           feature_names=["petal length", "petal width"], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png(path='tree.png', prog=None)


"""

"""
if __name__ == "__main__":
    pass
