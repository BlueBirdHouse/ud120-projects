#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )

#移除检查到的异常点
#1. 数据录入人员错误的将总数当作某一个人输入了数据库
data_dict.pop('TOTAL')

#2. 两个人和土匪一样， 其工资和奖金都超级高
for key in data_dict.keys():
    aSalary = data_dict[key]['salary']
    aBonu = data_dict[key]['bonus']
    if aSalary == "NaN":
        aSalary = 0
    if aBonu == "NaN":
        aBonu = 0 
    if aSalary > 1000000:
        if aBonu > 5000000:
            print(key)



features = ["bonus", "salary"]
data = featureFormat(data_dict, features,remove_any_zeroes=True)
bonus, salary = targetFeatureSplit( data )
#这里的features是矩阵组成的list，转换一下变成基本的list
for printer in range(len(salary)):
    salary[printer] = salary[printer][0]

### your code below
#绘图
#这里出现的超大数据(outlier),原因是数据录入人员错误的将总数当作某一个人输入了数据库
import matplotlib.pyplot as plt
plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


