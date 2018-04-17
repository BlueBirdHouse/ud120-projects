# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:38:28 2018

@author: Bird
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


#%% 读入数据
file_handler = open('your_word_data.pkl', "rb")
word_data = pickle.load(file_handler)
file_handler.close()


#%% in Part 4, do TfIdf vectorization here
#放在word_data_TFIDF文件内
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                                 stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english')
#Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
'''
When building the vocabulary ignore terms that have a document frequency 
strictly higher than the given threshold (corpus-specific stop words). 
If float, the parameter represents a proportion of documents, integer 
absolute counts. 
'''
word_data_tfidf = vectorizer.fit_transform(word_data)
feature_names = vectorizer.get_feature_names()