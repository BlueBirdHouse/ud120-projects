#!/usr/bin/python

"""
There are lots of finance features available, some of which might be more 
powerful than others in terms of predicting a person’s bonus. For example, 
suppose you thought about the data a bit and guess that the 
“long_term_incentive” feature, which is supposed to reward employees 
for contributing to the long-term health of the company, might be more closely 
related to a person’s bonus than their salary is. 

A way to confirm that you’re right in this hypothesis is to regress the bonus 
against the long term incentive, and see if the regression score is 
significantly higher than regressing the bonus against the salary. 
Perform the regression of bonus against long term incentive--what’s the score 
on the test data?
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "rb") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = '../tools/python2_lesson06_keys.pkl')
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)



### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()



#reg.score(feature_train,target_train)

#reg.score(feature_test,target_test)













