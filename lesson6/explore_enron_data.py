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
import numpy as np
import pandas as pd



enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# How many data points (people) are in the dataset?
print('Number of people in the Enron dataset: {0}'.format(len(enron_data)))
# output: 146

# For each person, how many features are available?
print('Number of features for each person in the Enron dataset: {0}'.format(len(enron_data.values()[0])))
# output: 21

# count the number of entries in the dictionary where data[person_name]["poi"]==1
pois = [x for x, y in enron_data.items() if y['poi']]
print('Number of POI\'s: {0}'.format(len(pois)))
# output: 18

'''
We compiled a list of all POI names (in ../final_project/poi_names.txt) 
and associated email addresses (in ../final_project/poi_email_addresses.py).
How many POI’s were there total? (Use the names file, not the email addresses, 
since many folks have more than one address and a few didn’t work for Enron, so we don’t have their emails.)

'''
# How many POI’s were there total?
# output:35

# What is the total value of the stock belonging to James Prentice?
enron_data['PRENTICE JAMES']['total_stock_value']
# output: 1095040

# How many email messages do we have from Wesley Colwell to persons of interest?
enron_data['PRENTICE JAMES']
enron_data['COLWELL WESLEY']['from_this_person_to_poi']
# output: 11

# What’s the value of stock options exercised by Jeffrey K Skilling?
enron_data['COLWELL WESLEY']
enron_data['SKILLING JEFFREY K']['exercised_stock_options']
# output: 19250000


enron_data['SKILLING JEFFREY K']['total_payments']
# output:8682716
enron_data['LAY KENNETH L']['total_payments']
# output:103559793
enron_data['FASTOW ANDREW S']['total_payments']
# output:2424083


# How many folks in this dataset have a quantified salary? What about a known email address?
count_salary = 0
count_email = 0
for key in enron_data.keys():
    if enron_data[key]['salary'] != 'NaN':
        count_salary+=1
    if enron_data[key]['email_address'] != 'NaN':
        count_email+=1
print(count_salary)
print(count_email)
# output: 95
# output: 111


# How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments? 
# What percentage of people in the dataset as a whole is this?
count_NaN_tp = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN':
        count_NaN_tp+=1
print(count_NaN_tp)
print(float(count_NaN_tp)/len(enron_data.keys()))
# output: 21, 0.143835616438
# 21 out of 146 (about 14%) of the people in the dataset don't have total_payments filled in.


# How many POIs in the E+F dataset have “NaN” for their total payments? 
# What percentage of POI’s as a whole is this?
count_NaN_tp = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] == True :
        print 
        count_NaN_tp+=1
print(count_NaN_tp)
print(float(count_NaN_tp)/len(enron_data.keys()))
# output:0, 0.0
# 0 out of 18, or 0% of POI's don't have total_payments filled.


# What is the new number of people of the dataset? 
# What is the new number of folks with “NaN” for total payments?
len(enron_data.keys())
# output:146
# Now there are 156 folks in dataset, 
# 31 of whom have "NaN" total_payments. 
# This makes for 20% of them with a "NaN" overall.

# What is the new number of POI’s in the dataset? :28
# What is the new number of POI’s with NaN for total_payments?: 10

# Now there are 28 POI's, 10 of whom have "NaN" for total_payments
# That's 36% of the POI's who have "NaN" for total_payments, a big jump from before.

