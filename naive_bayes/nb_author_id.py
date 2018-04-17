#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

classifier = GaussianNB()

t0 = time()
classifier.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

#当然是预测速度快了.预测比训练还慢,说明这种方法压根就没用.
t0 = time()
Temp = classifier.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

accuracy = classifier.score(features_test, labels_test)
print(accuracy)

#########################################################


