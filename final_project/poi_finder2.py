#!/usr/bin/python

'''
尝试完成UD120课程设计
第二阶段，将PIO的邮件数据合并在一起，将非PIO的邮件数据合并在一起。创造两个词频数据库然后分类。
这个脚本生成PIO和非PIO的邮件路径集合
'''

import sys
sys.path.append("../tools/")
from os import listdir

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


from help_Functions import data_Sort, featureEmail, featureFormat_Name
from feature_format import targetFeatureSplit,featureFormat

### Load the dictionary containing the dataset
with open("data_dict_Stage1.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

#%% 重新组成的数据集合
'''
features_list = ['poi','from_this_person_to_poi','to_messages']
data = featureFormat_Name(data_dict, features_list, sort_keys = True)
data = data_Sort(data,column = 0)
data = np.array(data)
'''

#%%调查PIO和非PIO的最少邮件数量
'''
    为了保证生成的混合数据库每一个人均拥有同样数量的邮件，这就需要每一个人拥有的邮件数量最多
    为所有人的最小值
'''
inFromPIO = []
inFromPIO_min = 10000000000
inFromPIOFiles = listdir('./fromPIO/')
for afile in inFromPIOFiles:
    aPIO = []
    aPIO.append(afile)
    afile = open('./fromPIO/' + afile, "r")
    afile.seek(0)
    aText = afile.read()
    aText = aText.splitlines()
    aPIO.append(len(aText))
    inFromPIO.append(aPIO)
    inFromPIO_min = np.min([inFromPIO_min,len(aText)])
    
inFromNoPIO = []
inFromNoPIO_min = 10000000000
inFromNoPIOFiles = listdir('./fromNoPIO/')
for afile in inFromNoPIOFiles:
    aNoPIO = []
    aNoPIO.append(afile)
    afile = open('./fromNoPIO/' + afile, "r")
    afile.seek(0)
    aText = afile.read()
    aText = aText.splitlines()
    aNoPIO.append(len(aText))
    inFromNoPIO.append(aNoPIO)
    inFromNoPIO_min = np.min([inFromNoPIO_min,len(aText)])
    
 
#%% 从每一个人拥有的邮件数量中随机抽取一定数量的邮件   
mailDir_PIO = []
number_Mail = inFromPIO_min - 1
for afile in inFromPIOFiles:
    afile = open('./fromPIO/' + afile, "r")
    afile.seek(0)
    aText = afile.read()
    aText = aText.splitlines()
    _,aText = train_test_split(aText,test_size = number_Mail,shuffle = True)
    mailDir_PIO = mailDir_PIO + aText

mailDir_NoPIO = []
number_Mail = inFromNoPIO_min - 1
for afile in inFromNoPIOFiles:
    afile = open('./fromNoPIO/' + afile, "r")
    afile.seek(0)
    aText = afile.read()
    aText = aText.splitlines()
    _,aText = train_test_split(aText,test_size = number_Mail,shuffle = True)
    mailDir_NoPIO = mailDir_NoPIO + aText    
    

#%% 存储需要处理的邮件信息
with open('data_dict_Stage2_1.pkl', "wb") as clf_outfile:
    pickle.dump(mailDir_PIO, clf_outfile,pickle.HIGHEST_PROTOCOL)
    
with open('data_dict_Stage2_2.pkl', "wb") as clf_outfile:
    pickle.dump(mailDir_NoPIO, clf_outfile,pickle.HIGHEST_PROTOCOL)
    
    
'''
from_benglisans  = open("./fromPIO/from_ben.glisan@enron.com.txt", "r")
from_benglisans.seek(0)
aText = from_benglisans.read()
aText = aText.splitlines()
'''







