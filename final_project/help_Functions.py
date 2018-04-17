# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:45:36 2018

@author: Bird

这里面包含用来帮助完成课程设计的文件.

"""
import numpy as np


def featureFormat_Name( dictionary, features, remove_NaN=False, remove_all_zeroes=False, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
        这个程序在产生数据列表的时候会带上人的名字。
        注意,numpy矩阵只要有一个元素是字符，则全部都是字符。所以这里改用list来存储数据。
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key "+ feature+ " not present")
                return
            value = dictionary[key][feature]
            
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )
        
        tmp_list.append(key)
        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( tmp_list )

    return return_list

def data_Sort(data,column = 0):
    #删除列表data里面的NaN并针对column一列排序
    for x_Counter in range(len(data)):
        for y_Counter in range(len(data[0])):
            datum = data[x_Counter][y_Counter]
            #print(datum)
            if np.isreal(datum):
                if np.isnan(datum):
                    data[x_Counter][y_Counter] = 0
                    
    data.sort(key = lambda x:x[column])
    return data
#numpy 排序是这样的
#data = data[data[:,1].argsort(),:]

def featureFormat_binary(data, features = ['director_fees','restricted_stock_deferred','total_stock_value','expenses','total_payments','other']):
    """ 
    生成逻辑feature
    data是featureFormat_Name生成的一个列表
    第一位的label，0表示肯定不是PIO，1表示可能是PIO但是不一定
    """
    outPut = np.zeros([len(data),len(features)+1])
    #开始填充数据
    
    #默认所有人都不一定是PIO
    outPut[:,0] = 1
    
    for x_pringter in range(len(data)):
        for y_printer in range(len(features)):
            if features[y_printer] == 'director_fees':
                datum = data[x_pringter][y_printer+1]
                if np.abs(datum) <= 1:
                    outPut[x_pringter,y_printer+1] = np.float(1)
                else:
                    outPut[x_pringter,0] = np.float(0)
            if features[y_printer] == 'restricted_stock_deferred':
                datum = data[x_pringter][y_printer+1]
                if np.abs(datum) <= 1:
                    outPut[x_pringter,y_printer+1] = np.float(1)
                else:
                    outPut[x_pringter,0] = np.float(0)
            if features[y_printer] == 'total_stock_value':
                datum = data[x_pringter][y_printer+1]
                if datum > 1:
                    outPut[x_pringter,y_printer+1] = np.float(1)
                else:
                    outPut[x_pringter,0] = np.float(0)        
            if features[y_printer] == 'expenses':
                datum = data[x_pringter][y_printer+1]
                if np.abs(datum) > 1:
                    outPut[x_pringter,y_printer+1] = np.float(1)
                else:
                    outPut[x_pringter,0] = np.float(0)        
            if features[y_printer] == 'total_payments':
                datum = data[x_pringter][y_printer+1]
                if np.abs(datum) > 1:
                    outPut[x_pringter,y_printer+1] = np.float(1)
                else:
                    outPut[x_pringter,0] = np.float(0)    
            if features[y_printer] == 'other':
                datum = data[x_pringter][y_printer+1]
                if np.abs(datum) > 1:
                    outPut[x_pringter,y_printer+1] = np.float(1)
                else:
                    outPut[x_pringter,0] = np.float(0)            
                    
    return outPut

def featureEmail( dictionary):
    '''
    这个函数输出数据库中的PIO和邮箱地址数据。
    '''
    outPut = []
    for aKey in dictionary.keys():
        aList = []
        aList.append(aKey)
        aList.append(dictionary[aKey]['poi'])
        aList.append(dictionary[aKey]['email_address'])
        outPut.append(aList)
    return outPut








