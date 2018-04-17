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


from help_Functions import featureFormat_Name, data_Sort,featureFormat_binary
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
data_binary = featureFormat_binary(data)
#data = np.array(data)
labels, features = targetFeatureSplit(data_binary)
labels = np.array(labels)
features = np.array(features)
#x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33,shuffle = True)
clf = AdaBoostClassifier(n_estimators=99)

kf = KFold(n_splits=4,shuffle=True)
for train_index, test_index in kf.split(features):
   x_train, x_test = features[train_index], features[test_index]
   y_train, y_test = labels[train_index], labels[test_index]
   
   clf.fit(x_train,y_train)
   pred = clf.predict(x_test)
   acc = accuracy_score(y_test,pred)
   print('对可能是PIO的判断准确程度是：'+str(acc))

#%%
print('现在利用上级分类器的决策，缩减数据库。')
pred = clf.predict(features)
print('这一步的准确程度是：')
acc = accuracy_score(labels,pred)
print(str(acc))
pop_Index = []
for counter in range(len(data)):
    if np.abs( pred[counter] ) < 0.1:
        pop_Index.append(counter)
data = [data[i] for i in pop_Index]
for datum in data:
    aName = datum[-1]
    data_dict.pop(aName)

#%% 存储修正过的数据库
with open('data_dict_Stage1.pkl', "wb") as clf_outfile:
    pickle.dump(data_dict, clf_outfile,pickle.HIGHEST_PROTOCOL)
#%% 重新组成的数据集合
'''
features_list = ['poi','from_messages','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi','to_messages']
data = featureFormat_Name(data_dict, features_list, sort_keys = True)
data = data_Sort(data,column = 3)
'''

