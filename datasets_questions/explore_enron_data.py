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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))


#How many POIs are there in the E+F dataset?
poiIn_enron_data = {}
for person in enron_data:
    if enron_data[person]['poi'] == 1:
        poiIn_enron_data[person] = enron_data[person]


'''
#Of these three individuals (Lay, Skilling and Fastow), who took home the most
# money (largest value of “total_payments” feature)?
print('LAY KENNETH L:')
print(poiIn_enron_data['LAY KENNETH L']['total_payments'])
print('SKILLING JEFFREY K:')
print(poiIn_enron_data['SKILLING JEFFREY K']['total_payments'])
print('FASTOW ANDREW S:')
print(poiIn_enron_data['FASTOW ANDREW S']['total_payments'])
'''

'''
#How many folks in this dataset have a quantified salary? What about a known
# email address? 
import numpy as np
quantifiedSalaryIn_enron_data = {}
knownEmailIn_enron_data = {}
for person in enron_data:
    if np.isreal(enron_data[person]['salary']) == np.True_:
        quantifiedSalaryIn_enron_data[person] = enron_data[person]
        
    if (enron_data[person]['email_address']).find('@') >= 0:
        knownEmailIn_enron_data[person] = enron_data[person]
'''

#How many people in the E+F dataset (as it currently exists) have “NaN” for 
#their total payments? What percentage of people in the dataset as a whole is this?
import numpy as np
totalPayments_NAN = {}
for person in enron_data:
    if np.isreal(enron_data[person]['total_payments']) == np.False_:
        totalPayments_NAN[person] = enron_data[person]
totalPayments_NAN_Per = len(totalPayments_NAN)/len(enron_data)

#How many POIs in the E+F dataset have “NaN” for their total payments? 
#What percentage of POI’s as a whole is this? 
totalPayments_NAN_POI = {}
for person in poiIn_enron_data:
    if np.isreal(poiIn_enron_data[person]['total_payments']) == np.False_:
        totalPayments_NAN_POI[person] = poiIn_enron_data[person]
totalPayments_NAN_POI_Per = len(totalPayments_NAN_POI)/len(poiIn_enron_data)










