#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
# Using the starter code in decision_tree/dt_author_id.py, 
# get a decision tree up and running as a classifier, setting min_samples_split=40. 
# It will probably take a while to train. 
# What’s the accuracy?
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test))
# output: 0.978384527873

#########################################################
# What's the number of features in your data?

print(len(features_train[0]))
# output: 3785
###########################################################################################
# Change percentile from 10 to 1, and rerun dt_author_id.py. What’s the number of features now?
features_train, features_test, labels_train, labels_test = preprocess(percentile=1)
clf2 = tree.DecisionTreeClassifier(min_samples_split=40)
clf2 = clf2.fit(features_train, labels_train)
print(len(features_train[0]))
# output: 379 features with 1 percentile
######################################################################################

# What's the accuracy of your decision tree when you use only 1% of your available features 
# (i.e. percentile=1)?
print(clf2.score(features_test, labels_test))
#output: 0.967007963595
#########################################################


