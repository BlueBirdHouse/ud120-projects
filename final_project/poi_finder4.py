#!/usr/bin/python

'''
尝试完成UD120课程设计
第四阶段，将PIO的邮件数据合并在一起，将非PIO的邮件数据合并在一起。创造两个词频数据库然后分类。

'''

import sys
sys.path.append("../tools/")
from os import listdir

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import KFold
#import tarfile
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
#from sklearn import svm
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#from help_Functions import data_Sort, featureEmail, featureFormat_Name
#from feature_format import targetFeatureSplit,featureFormat
#from parse_out_email_text import parseOutText



#%% Load the dictionary containing the dataset
with open("data_dict_Stage3_1.pkl", "rb") as data_file:
    word_data = pickle.load(data_file)
    
with open("data_dict_Stage3_2.pkl", "rb") as data_file:
    authors = pickle.load(data_file)

#%% 删除签名feature
for counter in range(len(word_data)):
    print('正在处理：'+ str(counter))
    atext = word_data[counter]
    atext = atext.replace('kenneth','')
    atext = atext.replace('ben','')
    atext = atext.replace('johnson','')
    atext = atext.replace('ann','')
    atext = atext.replace('sherri','')
    atext = atext.replace('paula','')
    atext = atext.replace('ken','')
    atext = atext.replace('hoffman','')
    atext = atext.replace('chris','')
    atext = atext.replace('janic','')
    atext = atext.replace('kaufman','')
    atext = atext.replace('julia','')
    atext = atext.replace('rieker','')
    atext = atext.replace('kevin','')
    atext = atext.replace('smith','')
    atext = atext.replace('steve','')
    atext = atext.replace('miller','')
    atext = atext.replace('calpin','')
    atext = atext.replace('mechell','')
    atext = atext.replace('kelli','')
    atext = atext.replace('andrew','')
    atext = atext.replace('colwel','')
    atext = atext.replace('jonathan','')
    atext = atext.replace('david','')
    atext = atext.replace('justin','')
    atext = atext.replace('vicki','')
    atext = atext.replace('crisi','')
    atext = atext.replace('theresa','')
    atext = atext.replace('ray','')
    atext = atext.replace('cohen','')
    atext = atext.replace('coyn','')
    atext = atext.replace('shelbi','')
    atext = atext.replace('wes','')
    atext = atext.replace('rex','')
    atext = atext.replace('tim','')
    atext = atext.replace('dyer','')
    atext = atext.replace('welch','')
    atext = atext.replace('blanchard','')
    atext = atext.replace('promin','')
    atext = atext.replace('thomson','')
    atext = atext.replace('williamson','')
    atext = atext.replace('rei','')
    atext = atext.replace('palmer','')
    atext = atext.replace('robert','')
    atext = atext.replace('brian','')
    atext = atext.replace('mond','')
    atext = atext.replace('rodney','')
    atext = atext.replace('gordon','')
    atext = atext.replace('martin','')
    atext = atext.replace('mike','')
    atext = atext.replace('joseph','')
    atext = atext.replace('tom','')
    atext = atext.replace('rick','')
    atext = atext.replace('ron','')
    atext = atext.replace('freeman','')
    atext = atext.replace('joe','')
    atext = atext.replace('jeffrey','')
    atext = atext.replace('agre','')
    atext = atext.replace('cox','')
    
    #消除删除文字以后产生的多于空格
    atext = atext.split()
    atext = ' '.join(atext)
    word_data[counter] = atext

#%% 开始做词频统计
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.2, shuffle=True)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.9,
                             stop_words='english',min_df = 0.01)
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test)

#%% 选择主要的feature

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train).toarray()
features_test  = selector.transform(features_test).toarray()

#%% 寻找签名feature
importances = selector.scores_
num = features_train.shape[1]
index_sort = importances.argsort(axis = 0)[-num:][::-1]  
print('Feature重要性得分为：')
print(importances[index_sort[0:(num-1)]])
feature_names = vectorizer.get_feature_names()
for counter in range(num):
    print(feature_names[index_sort[counter]])


#%% 生成分类器分类
#clf = svm.SVC(kernel='linear')
clf_base = GaussianNB()
#clf = DecisionTreeClassifier()
clf = AdaBoostClassifier(n_estimators=99,base_estimator = clf_base)
clf.fit(features_train, labels_train)  

#%% 分类器评分
pred = clf.predict(features_test)


acc = accuracy_score(labels_test,pred)
print('第二步骤的accuracy 得分是：' + str(acc) )
precisionScore = precision_score(labels_test, pred)
print("第二步骤的precision 指标是:" + str(precisionScore) )
recallScore = recall_score(labels_test, pred)
print('第二步骤的recall 指标是：' + str(recallScore) )

#%% 计算其他参数
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for prediction, truth in zip(pred, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    elif prediction == 1 and truth == 1:
        true_positives += 1
    else:
        print("Warning: Found a predicted label not == 0 or 1.")
        print("All predictions should take value 0 or 1.")
        print("Evaluating performance for processed predictions:")
        break
    
total_predictions = true_negatives + false_negatives + false_positives + true_positives
print('两步合并计算的准确性是：')
total_accuracy = 1.0*(true_positives + true_negatives + 77)/(total_predictions + 77)
print(str(total_accuracy))

print('第二步骤的f1和f2指标是：')
f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
f2 = (1+2.0*2.0) * precisionScore*recallScore/(4*precisionScore + recallScore)
print(str(f1))
print(str(f2))



