# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:11:30 2018

@author: Bird
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

#每个树的最大深度 max_depth
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)


#数值越高，这个特征的重要性越高
print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))
