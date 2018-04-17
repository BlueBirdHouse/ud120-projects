#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
#这个部分将课程所提问的问题写成函数


import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

#How many POIs are there in the E+F dataset?
def poiIn_enron(enron_data = enron_data):
    poiIn_enron_data = {}
    for person in enron_data:
        if enron_data[person]['poi'] == 1:
            poiIn_enron_data[person] = enron_data[person]
    return poiIn_enron_data

