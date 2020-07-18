# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei CHEN
Email:cxw19@mails.tsinghua.edu.cn
"""

"""
对收到较大方差影响的多个决策树取平均值， 以建立一个更好的繁华性能和不易过拟合的模型
步骤：
    1.随机提取规模为n的引导样本（训练集中随机选取n个***【可替换】***样本）
    2.基于引导样本的数据声称决策树。在每个节点：
        a.【随机选择d个特征】，无需替换    ******* d = m **.5, m = num of features
        b.根据目标函数提供的最佳分裂特征来分裂节点，如最大化信息增益
    3.把步骤1，2重复k次
    聚合每棵树的预测结果，以多数票机制确定标签分类
    
通常，树越多，随机森林分类器的性能越好，当然计算成本的增加就越大
缩小导引样本规模，可能增加随机森林的随机性，减少过拟合的影响；但会导致总体性能较差
"""

from sklearn.ensemble import RandomForestClassifier
from Perceptron import X_train, y_train


# n_estimator 为25颗决策树，拥护熵作为杂质度判断准则来分裂节点， n_jobs 为多核计算机并行训练
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)



if __name__ == "__main__":
    pass
