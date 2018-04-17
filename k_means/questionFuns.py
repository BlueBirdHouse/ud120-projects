# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:14:38 2018

@author: Bird
"""

def dataList(dictionary,aFeature):
    #这个函数用来回答问题
    '''
    What are the maximum and minimum values taken by the 
    “exercised_stock_options” feature used in this example?
    '''
    #它仅提取dictionary目标为aFeature键的内容，然后输出。
    #由于字典是无序访问的，所以内容是随机的
    
    allKeys = dictionary.keys()
    outPut = []
    for aKey in allKeys:
        datum = dictionary[aKey][aFeature]
        if datum != "NaN":
            outPut.append(datum)
    return outPut