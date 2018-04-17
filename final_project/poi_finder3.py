#!/usr/bin/python

'''
尝试完成UD120课程设计
第三阶段，将PIO的邮件数据合并在一起，将非PIO的邮件数据合并在一起。创造两个词频数据库然后分类。
这个脚本读取PIO和非PIO的邮件，取出词干以后加以存储
'''

import sys
sys.path.append("../tools/")
#from os import listdir

import pickle
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import KFold
import tarfile
#from nltk.stem.snowball import SnowballStemmer


#from help_Functions import data_Sort, featureEmail, featureFormat_Name
#from feature_format import targetFeatureSplit,featureFormat
from parse_out_email_text import parseOutText
#%% Load the dictionary containing the dataset
with open("data_dict_Stage2_1.pkl", "rb") as data_file:
    mailDir_PIO = pickle.load(data_file)
    
with open("data_dict_Stage2_2.pkl", "rb") as data_file:
    mailDir_NoPIO = pickle.load(data_file)

#%% 做词频统计
tar = tarfile.open('../maildir/enron_mail_20150507.tar.gz','r:gz')
#stemmer = SnowballStemmer("english", ignore_stopwords=False)

temp_counter = 0
from_data = []
word_data = []

for name, from_person in [("PIO", mailDir_PIO), ("NoPIO", mailDir_NoPIO)]:
    for path in from_person:
        #path = path[:-1]
        path = 'maildir' + path.split('maildir')[1]
        temp_counter = temp_counter + 1
        print(temp_counter)
        print(path)  
       
        email = tar.extractfile(path)
        
        #找到每一个词的词干
        ### use parseOutText to extract the text from the opened email
        atext = parseOutText(email,working = True)
        #print(atext)
        
        #删除那些明显带有署名标记的词语。 用来检验算法的真确性。
        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]
        # if you can figure out why it's "germani" and not "germany")
        # 因为转换词干的过程中，会将germany转换成germani。 更可靠的方法是使用下面的代码。
        #atext = atext.replace(stemmer.stem('sara'),'')
        #atext = atext.replace(stemmer.stem('shackleton'),'')
        #atext = atext.replace(stemmer.stem('chris'),'')
        #atext = atext.replace(stemmer.stem('germany'),'')
        #消除删除文字以后产生的多于空格
        atext = atext.split()
        atext = ' '.join(atext)
            
        ### append the text to word_data
        word_data.append(atext)
        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == "NoPIO":
            from_data.append(0)
        elif name == "PIO":
            from_data.append(1)
        else:
            raise NameError('不知道这个是啥类！')
                
        email.close()
   
 #%% 关闭压缩包文件
tar.close()   

#%% 文件存储
with open("data_dict_Stage3_1.pkl", 'wb') as pfile:
    pickle.dump(word_data, pfile, pickle.HIGHEST_PROTOCOL)
with open("data_dict_Stage3_2.pkl", 'wb') as pfile:
    pickle.dump(from_data, pfile, pickle.HIGHEST_PROTOCOL)
    
    
    
    