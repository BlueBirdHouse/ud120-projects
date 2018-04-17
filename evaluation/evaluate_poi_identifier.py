#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)
print('从数据集里面分一部分出来以后的准确性：')
print(str(acc))

#%%
'''
Precision and recall can help illuminate your performance better. Use the 
precision_score and recall_score available in sklearn.metrics to compute those 
quantities.
'''
from sklearn.metrics import precision_score
precisionScore = precision_score(labels_test, pred)
print('准确性指标是：%d' %precisionScore )
from sklearn.metrics import recall_score
recallScore = recall_score(labels_test, pred)
print('回忆性指标是：%d' %recallScore )




