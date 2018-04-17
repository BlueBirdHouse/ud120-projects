#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###


#clf = svm.SVC(kernel='linear')
clf = svm.SVC(kernel='rbf',C=10000)

#%% 下面的代码缩减训练数据以测试对准确度和训练速度的影响
# 注意，最后一个练习一定不能缩减训练集，否则答案不对。
#features_train = features_train[:round(len(features_train)/100)]
#labels_train = labels_train[:round(len(labels_train)/100)]
#非常有意的是预测时间也减少了
#%% 

t0 = time()
clf.fit(features_train, labels_train)  
print("training time:", round(time()-t0, 3), "s")

#%%
t0 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")
acc = accuracy_score(labels_test,pred)
#%%

#########################################################


