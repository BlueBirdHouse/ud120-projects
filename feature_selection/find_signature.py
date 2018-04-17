#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )

#由于重新运行‘vectorize_text_ReadGz.py’太费时间了，这里尝试在数据上处理那些
# ‘signature words’
for counter in range(len(word_data)):
    print('正在处理：'+ str(counter))
    atext = word_data[counter]
    atext = atext.replace('sshacklensf','')
    atext = atext.replace('cgermannsf','')
    #消除删除文字以后产生的多于空格
    atext = atext.split()
    atext = ' '.join(atext)
    word_data[counter] = atext

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_train = features_train.toarray()
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]

### your code goes here
#生成决策树
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42,min_samples_split=50)
clf.fit(features_train,labels_train)
from sklearn.metrics import accuracy_score
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)
print('当前决策树的准确程度是：'+str(acc))

#%% 测试的结果显示,即便是用很少的样本点, 结果也很好, 说明学习数据里面有很强的特征.
#这可能是编程问题
'''
What’s the importance of the most important feature? 
What is the number of this feature? 
'''
#将重要性从大到小排列
importances = clf.feature_importances_
num = 10
index_sort = importances.argsort(axis = 0)[-num:][::-1]  
print(index_sort)
print(importances[index_sort[0]])

#%% pull out the word that’s causing most of the discrimination of the 
# decision tree.
feature_names = vectorizer.get_feature_names()
print(feature_names[index_sort[0]])
'''
结果显示，有标记‘sshacklensf’的，基本上一定是PIO, 这就不正常。难道是PIO自己给自己的标记吗？
'''

#%% 





