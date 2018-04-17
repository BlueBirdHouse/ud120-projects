#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)


### it's all yours from here forward!  

'''
Create a decision tree classifier (just use the default parameters), train it 
on all the data (you will fix this in the next part!), 
and print out the accuracy. 
'''
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features,labels)
pred = clf.predict(features)
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels,pred)
print('使用整个数据集训练以后的准确性：')
print(str(acc))


'''
hold out 30% of the data for testing and set the random_state parameter to 42 
(random_state controls which points go into the training set and which are used
 for testing; setting it to 42 means we know exactly which events are in which
 set, and can check the results you get). What’s your updated accuracy?
'''
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)
print('从数据集里面分一部分出来以后的准确性：')
print(str(acc))




