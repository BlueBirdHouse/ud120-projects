#!/usr/bin/python

'''
尝试完成UD120课程设计
第一阶段，从数据库中删除那些明显不是PIO的人。
有些人与PIO有明显的差别，这些人在数据库里面可能会导致混乱
'''

import sys
sys.path.append("../tools/")

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from help_Functions import featureFormat_Name, data_Sort
from feature_format import targetFeatureSplit




#features_list = ['poi','salary'] # You will need to use more features
#features_list = ['poi','director_fees'] 
#features_list = ['poi','exercised_stock_options'] 
#features_list = ['poi','restricted_stock'] 
#features_list = ['poi','restricted_stock_deferred'] 
#features_list = ['poi','total_payments'] 
#features_list = ['poi','bonus'] 
#features_list = ['poi','deferred_income'] 
#features_list = ['poi','deferral_payments'] 
#features_list = ['poi','expenses'] 
#features_list = ['poi','total_payments'] 
#features_list = ['poi','other'] 
#features_list = ['poi','long_term_incentive'] 
#features_list = ['poi','loan_advances'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
#这个outlier是一定要移除的。
data_dict.pop('TOTAL')

#%% 使用二进制feature剔除肯定不是PIO的无辜者
features_list = ['poi','director_fees','restricted_stock_deferred','total_stock_value','expenses','total_payments','other']
### Extract features and labels from dataset for local testing
data = featureFormat_Name(data_dict, features_list, sort_keys = True)
#按照PIO排序
data = data_Sort(data,column = 0)
data = np.array(data)
data = data[:,0:6]
data = data.astype(np.float)
data = np.abs(data)
labels, features = targetFeatureSplit(data)
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)

labels = np.array(labels)
#x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33,shuffle = True)
clf = AdaBoostClassifier(n_estimators=99)

kf = KFold(n_splits=4,shuffle=True)
for train_index, test_index in kf.split(features):
   x_train, x_test = features[train_index], features[test_index]
   y_train, y_test = labels[train_index], labels[test_index]
   
   clf.fit(x_train,y_train)
   pred = clf.predict(x_test)
   
   print('-------------')
   acc = accuracy_score(y_test,pred)
   print('对肯定不是PIO的判断准确程度是：'+str(acc))
   precisionScore = precision_score(y_test, pred,pos_label=0)
   print("对肯定不是PIO的precision 指标是:" + str(precisionScore) )
   recallScore = recall_score(y_test, pred,pos_label=0)
   print('对肯定不是PIO的recall 指标是：' + str(recallScore) )
