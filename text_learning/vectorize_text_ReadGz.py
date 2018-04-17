#!/usr/bin/python

import os
import pickle
import re
import sys
import tarfile
from nltk.stem.snowball import SnowballStemmer


sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""
tar = tarfile.open('../maildir/enron_mail_20150507.tar.gz','r:gz')

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

stemmer = SnowballStemmer("english", ignore_stopwords=False)
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        #if temp_counter < 200:
        if True:
            #path = os.path.join('..', path[:-1])
            path = path[:-1]
            print(temp_counter)
            print(path)
            #email = open(path, "r")
            #在线解压文件
            email = tar.extractfile(path)

            #找到每一个词的词干
            ### use parseOutText to extract the text from the opened email
            atext = parseOutText(email,working = True)
            
            #删除那些明显带有署名标记的词语。 用来检验算法的真确性。
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            # if you can figure out why it's "germani" and not "germany")
            # 因为转换词干的过程中，会将germany转换成germani。 更可靠的方法是使用下面的代码。
            atext = atext.replace(stemmer.stem('sara'),'')
            atext = atext.replace(stemmer.stem('shackleton'),'')
            atext = atext.replace(stemmer.stem('chris'),'')
            atext = atext.replace(stemmer.stem('germany'),'')
            #消除删除文字以后产生的多于空格
            atext = atext.split()
            atext = ' '.join(atext)
            
            ### append the text to word_data
            word_data.append(atext)
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            elif name == "chris":
                from_data.append(1)
            else:
                raise NameError('人名没有指定！')
                
            email.close()

print("emails processed")
from_sara.close()
from_chris.close()

#%% 文件存储
with open("your_word_data.pkl", 'wb') as pfile:
    pickle.dump(word_data, pfile, pickle.HIGHEST_PROTOCOL)
with open("your_email_authors.pkl", 'wb') as pfile:
    pickle.dump(from_data, pfile, pickle.HIGHEST_PROTOCOL)

#%% 关闭压缩包文件
tar.close()

### in Part 4, do TfIdf vectorization here
#放在word_data_TFIDF文件内














