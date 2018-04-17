#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.metrics import accuracy_score

#%% 使用k-邻居方法
print('使用k-邻居方法')
from sklearn import neighbors
clf_KNearest = neighbors.KNeighborsClassifier()
clf_KNearest.fit(features_train, labels_train)
pred = clf_KNearest.predict(features_test)

acc_KNearest = accuracy_score(labels_test,pred)
print(acc_KNearest)
plt.figure()
prettyPicture(clf_KNearest, features_test, labels_test)
plt.show()

#%% 使用随机森林方法
print('使用随机森林方法')
from sklearn.ensemble import RandomForestClassifier
clf_Forest = RandomForestClassifier()
clf_Forest.fit(features_train, labels_train)
pred = clf_Forest.predict(features_test)

acc_Forest = accuracy_score(labels_test,pred)
print(acc_Forest)
plt.figure()
prettyPicture(clf_Forest, features_test, labels_test)
plt.show()

#%% 使用AdaBoost方法
print('使用AdaBoost方法')
from sklearn.ensemble import AdaBoostClassifier
clf_Boost = AdaBoostClassifier()
clf_Boost.fit(features_train, labels_train)
pred = clf_Boost.predict(features_test)

acc_Boost = accuracy_score(labels_test,pred)
print(acc_Boost)
plt.figure()
prettyPicture(clf_Boost, features_test, labels_test)
plt.show()










