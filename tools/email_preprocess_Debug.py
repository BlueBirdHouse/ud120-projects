#!/usr/bin/python

#这个函数将存储的数据分为测试和训练两个部分。
#返回的数据每一行表示一封邮件，每一列表示每个单词的重要程度，及Tfidf值

import pickle
#import cPickle
import numpy as np

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



words_file = "../tools/word_data.pkl"
authors_file="../tools/email_authors.pkl"
""" 
    this function takes a pre-made list of email texts (by default word_data.pkl)
    and the corresponding authors (by default email_authors.pkl) and performs
    a number of preprocessing steps:
        -- splits into training/testing sets (10% testing)
        -- vectorizes into tfidf matrix
        -- selects/keeps most helpful features

    after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

    4 objects are returned:
        -- training/testing features
        -- training/testing labels

"""

### the words (features) and authors (labels), already largely preprocessed
### this preprocessing will be repeated in the text learning mini-project
authors_file_handler = open(authors_file, "rb")
authors = pickle.load(authors_file_handler)
authors_file_handler.close()

words_file_handler = open(words_file, "rb")
#word_data = cPickle.load(words_file_handler)
word_data = pickle.load(words_file_handler)
words_file_handler.close()

### test_size is the percentage of events assigned to the test set
### (remainder go into training)
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)



### text vectorization--go from strings to lists of numbers
#Tfid：词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
#这个矩阵，每一行都表示s字典里对应的一句话。每一列是对应的词的Tfid值。

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)

#找到表现最佳的10%特征，以降低运算量
### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result
# SelectPercentile返回表现最佳的前r%个特征。
# 如果不这样做，feature的个数会高达37851
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)

features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()

# info on the data
print("no. of Chris training emails:" + str(sum(labels_train)))
print("no. of Sara training emails:", str(len(labels_train)-sum(labels_train)))


