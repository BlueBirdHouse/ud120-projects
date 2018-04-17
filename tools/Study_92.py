# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:28:20 2018
为了学习原作者的设计目的，参考书籍《Python数据挖掘入门与实践》
92 (106 / 251)
当中的例子
@author: Bird
"""

s = """Three Rings for the Elven-kings under the sky
Seven for the Dwarf-lords in halls of stone
Nine for Mortal Men doomed to die
One for the Dark Lord on his dark throne
In the Land of Mordor where the Shadows lie
One Ring to rule them all One Ring to find them
One Ring to bring them all and in the darkness bind them
In the Land of Mordor where the Shadows lie """.lower()

words = s.split()

from collections import Counter

c = Counter(words)

#print(c.most_common(10))

#这个矩阵，每一行都表示s字典里对应的一句话。每一列是对应的词的Tfid值。

s = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

vectorizer.fit_transform(s)

print(vectorizer.get_feature_names())

tfidf_Array = vectorizer.fit_transform(s).toarray()

